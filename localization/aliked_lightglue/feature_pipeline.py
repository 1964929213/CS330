from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List

from localization.aliked_lightglue.image_io import read_image

DEFAULT_PROXY = ""


def set_proxy(proxy_url: str = DEFAULT_PROXY) -> None:
    if not proxy_url:
        return
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ[key] = proxy_url


class ImageDataset:
    def __init__(
        self, root: Path, conf: Dict, names: List[str], default_conf: Dict, resize_fn
    ):
        self.root = root
        self.conf = {**default_conf, **conf}
        self.names = list(names)
        self.resize_fn = resize_fn
        for name in self.names:
            if not (self.root / name).exists():
                raise FileNotFoundError(self.root / name)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        import numpy as np

        name = self.names[idx]
        image = read_image(self.root / name, self.conf["grayscale"]).astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf["resize_max"] and (
            self.conf.get("resize_force", False) or max(size) > self.conf["resize_max"]
        ):
            scale = self.conf["resize_max"] / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = self.resize_fn(image, size_new, self.conf["interpolation"])

        if self.conf["grayscale"]:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))

        return {"image": image / 255.0, "original_size": np.array(size)}


def extract_features(
    conf: Dict,
    image_dir: Path,
    image_list: List[str],
    feature_path: Path,
    overwrite: bool,
    as_half: bool,
) -> Path:
    from hloc import extract_features as hloc_extract, logger
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.io import list_h5_names
    from tqdm import tqdm
    import h5py
    import numpy as np
    import torch

    logger.info("Extracting features: %s", conf["output"])
    dataset = ImageDataset(
        image_dir,
        conf["preprocessing"],
        image_list,
        hloc_extract.ImageDataset.default_conf,
        hloc_extract.resize_image,
    )
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    dataset.names = [name for name in dataset.names if name not in skip]
    if not dataset.names:
        return feature_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = dynamic_load(hloc_extract.extractors, conf["model"]["name"])
    model = model_cls(conf["model"]).eval().to(device)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, shuffle=False, pin_memory=True
    )

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader)):
            name = dataset.names[idx]
            pred = model({"image": data["image"].to(device, non_blocking=True)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            pred["image_size"] = original_size = data["original_size"][0].numpy()

            if "keypoints" in pred:
                size = np.array(data["image"].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
                if "scales" in pred:
                    pred["scales"] *= scales.mean()
                uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

            if as_half:
                for key, value in pred.items():
                    if value.dtype == np.float32:
                        pred[key] = value.astype(np.float16)

            with h5py.File(str(feature_path), "a", libver="latest") as fd:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for key, value in pred.items():
                    grp.create_dataset(key, data=value)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
    return feature_path


class WorkQueue:
    def __init__(self, work_fn):
        self.queue = Queue(1)
        self.thread = Thread(target=self._run, args=(work_fn,))
        self.thread.start()

    def _run(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, item):
        self.queue.put(item)

    def join(self):
        self.queue.put(None)
        self.thread.join()


def match_features(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    import torch
    from tqdm import tqdm
    from hloc import logger, match_features as hloc_match
    from hloc.match_features import (
        FeaturePairsDataset,
        writer_fn,
        find_unique_new_pairs,
    )
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.parsers import names_to_pair, parse_retrieval

    logger.info("Matching features: %s", conf["output"])
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, refs in pairs.items() for r in refs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if not pairs:
        return match_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = dynamic_load(hloc_match.matchers, conf["model"]["name"])
    model = model_cls(conf["model"]).eval().to(device)
    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True
    )
    writer = WorkQueue(partial(writer_fn, match_path=match_path))

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
            data = {
                k: v if k.startswith("image") else v.to(device, non_blocking=True)
                for k, v in data.items()
            }
            writer.put((names_to_pair(*pairs[idx]), model(data)))
    writer.join()
    return match_path
