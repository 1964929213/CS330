from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

try:  # Support both package and direct script execution.
    from .config import load_config
    from .localization_utils import (
        compute_weighted_topometric_position,
        estimate_heading_with_pnp,
        get_query_camera,
        load_transform_2x3,
        read_reference_points,
    )
    from .schemas import PackageConfig
except ImportError:  # pragma: no cover - exercised by direct script launch.
    from config import load_config
    from localization_utils import (
        compute_weighted_topometric_position,
        estimate_heading_with_pnp,
        get_query_camera,
        load_transform_2x3,
        read_reference_points,
    )
    from schemas import PackageConfig


def compute_candidate_dispersion_px(
    used_candidates: list[str],
    weights: dict[str, float],
    ref_xy: dict[str, tuple[float, float]],
    center_xy: tuple[float, float] | None,
) -> float | None:
    if center_xy is None or not used_candidates:
        return None
    total_w = 0.0
    acc = 0.0
    cx, cy = float(center_xy[0]), float(center_xy[1])
    for name in used_candidates:
        xy = ref_xy.get(name)
        if xy is None:
            continue
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        dx = float(xy[0]) - cx
        dy = float(xy[1]) - cy
        acc += w * math.hypot(dx, dy)
        total_w += w
    if total_w <= 0:
        return None
    return acc / total_w


class Localizer:
    """In-memory ALIKED + LightGlue localizer.

    Runtime contract:
    - retrieve against every image key in the reference NetVLAD H5 file;
    - compute final floorplan x/y from weighted topometric candidates;
    - run PnP only for heading and diagnostic fields;
    - allow topometric-only reference images that are absent from the SfM model.
    """

    def __init__(self, config: PackageConfig | None = None):
        self.config = config or load_config()
        self.paths = self.config.paths
        self.runtime = self.config.runtime

        missing = [k for k, v in self.paths.required_path_map().items() if not v.exists()]
        if missing:
            raise RuntimeError(f'missing required localization inputs: {missing}')

        import h5py
        import numpy as np
        import pycolmap
        import torch
        from hloc import extract_features, match_features
        from hloc.utils.base_model import dynamic_load

        self.h5py = h5py
        self.np = np
        self.torch = torch
        self.extract_features = extract_features
        self.match_features = match_features

        self.reconstruction = pycolmap.Reconstruction(self.paths.sfm_model_dir)
        self.ref_xy = read_reference_points(self.paths.ref_points_2d)
        self.T_2x3 = load_transform_2x3(self.paths.transform_json)
        self.db_global_feats = self.paths.db_global_feats
        self.db_local_feats = self.paths.db_local_feats
        self.sfm_model_dir = self.paths.sfm_model_dir
        self.db_names = self.list_h5_image_names(self.db_global_feats)

        self.netvlad_conf = copy.deepcopy(extract_features.confs['netvlad'])
        self.local_conf = copy.deepcopy(extract_features.confs['aliked-n16'])
        self.matcher_conf = copy.deepcopy(match_features.confs['aliked+lightglue'])
        self.query_resize_max = self.runtime.query_resize_max
        self.netvlad_conf['preprocessing']['resize_max'] = self.query_resize_max
        self.local_conf['preprocessing']['resize_max'] = self.query_resize_max

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        NetvladModel = dynamic_load(extract_features.extractors, self.netvlad_conf['model']['name'])
        LocalModel = dynamic_load(extract_features.extractors, self.local_conf['model']['name'])
        MatcherModel = dynamic_load(match_features.matchers, self.matcher_conf['model']['name'])
        self.netvlad_model = NetvladModel(self.netvlad_conf['model']).eval().to(device)
        self.local_model = LocalModel(self.local_conf['model']).eval().to(device)
        self.matcher_model = MatcherModel(self.matcher_conf['model']).eval().to(device)
        self.db_global_desc = self.load_global_descriptors(self.db_global_feats, self.db_names)
        self.last_candidates: list[str] = []

    def resize_image_for_hloc(self, image, size, interp):
        return self.extract_features.resize_image(image, size, interp)

    def preprocess_single_image(self, path: Path, preprocessing: dict):
        from hloc.utils.io import read_image

        np = self.np
        image = read_image(path, preprocessing.get('grayscale', False)).astype(np.float32)
        size = image.shape[:2][::-1]
        resize_max = preprocessing.get('resize_max')
        resize_force = preprocessing.get('resize_force', False)
        if resize_max and (resize_force or max(size) > resize_max):
            scale = resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = self.resize_image_for_hloc(image, size_new, preprocessing.get('interpolation', 'cv2_area'))
        if preprocessing.get('grayscale', False):
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))
        image = image / 255.0
        return image, np.array(size)

    def extract_features_to_h5(self, model, conf: dict, image_path: Path, image_name: str, feature_path: Path, as_half: bool = True):
        np = self.np
        torch = self.torch
        h5py = self.h5py
        image, original_size = self.preprocess_single_image(image_path, conf['preprocessing'])
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.from_numpy(image)[None].to(device, non_blocking=True)
        with torch.no_grad():
            pred = model({'image': tensor})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = original_size
        if 'keypoints' in pred:
            size = np.array(image.shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + 0.5) * scales[None] - 0.5
            if 'scales' in pred:
                pred['scales'] *= scales.mean()
            uncertainty = getattr(model, 'detection_noise', 1) * scales.mean()
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(feature_path), 'a', libver='latest') as fd:
            if image_name in fd:
                del fd[image_name]
            grp = fd.create_group(image_name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
            if 'keypoints' in pred:
                grp['keypoints'].attrs['uncertainty'] = uncertainty

    def load_feature_group(self, path: Path, name: str):
        torch = self.torch
        data = {}
        with self.h5py.File(str(path), 'r', libver='latest') as fd:
            grp = fd[name]
            for k, v in grp.items():
                data[k] = torch.from_numpy(v.__array__()).float()
        return data

    def list_h5_image_names(self, path: Path) -> list[str]:
        with self.h5py.File(str(path), 'r', libver='latest') as fd:
            return sorted(fd.keys())

    def load_global_descriptors(self, path: Path, names: list[str]):
        np = self.np
        torch = self.torch
        with self.h5py.File(str(path), 'r', libver='latest') as fd:
            desc = [fd[n]['global_descriptor'].__array__() for n in names]
        return torch.from_numpy(np.stack(desc, 0)).float()

    def retrieve_topk(self, query_feature_path: Path, query_name: str, k: int) -> list[str]:
        torch = self.torch
        with self.h5py.File(str(query_feature_path), 'r', libver='latest') as fd:
            query_desc = torch.from_numpy(fd[query_name]['global_descriptor'].__array__()[None]).float()
        query_desc = query_desc.to(self.device)
        db_desc = self.db_global_desc.to(self.device)
        sim = torch.einsum('id,jd->ij', query_desc, db_desc)
        topk = torch.topk(sim, min(k, sim.shape[1]), dim=1, largest=True).indices[0].cpu().tolist()
        return [self.db_names[i] for i in topk]

    def match_pairs_to_h5(self, query_name: str, ref_names: list[str], query_feature_path: Path, ref_feature_path: Path, match_path: Path):
        torch = self.torch
        match_path.parent.mkdir(parents=True, exist_ok=True)
        q = self.load_feature_group(query_feature_path, query_name)
        with self.h5py.File(str(match_path), 'a', libver='latest') as fd:
            for ref_name in ref_names:
                r = self.load_feature_group(ref_feature_path, ref_name)
                image0_size = tuple(int(x) for x in q['image_size'].numpy()[::-1])
                image1_size = tuple(int(x) for x in r['image_size'].numpy()[::-1])
                data = {
                    'image0': torch.empty((1,) + image0_size),
                    'image1': torch.empty((1,) + image1_size),
                }
                for k, v in q.items():
                    if k == 'image_size':
                        continue
                    data[k + '0'] = v[None].to(self.device, non_blocking=True)
                for k, v in r.items():
                    if k == 'image_size':
                        continue
                    data[k + '1'] = v[None].to(self.device, non_blocking=True)
                with torch.no_grad():
                    pred = self.matcher_model(data)
                pair = self.match_features.names_to_pair(query_name, ref_name)
                if pair in fd:
                    del fd[pair]
                grp = fd.create_group(pair)
                matches = pred['matches0'][0].cpu().short().numpy()
                grp.create_dataset('matches0', data=matches)
                scores = pred['matching_scores0'][0].cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)

    def localize(self, query_image: Path) -> dict[str, Any]:
        from hloc.utils.io import get_matches

        run_dir = self.paths.work_dir / 'live'
        run_dir.mkdir(parents=True, exist_ok=True)
        query_name = query_image.name

        query_global_h5 = run_dir / 'query-global-feats-netvlad.h5'
        query_local_h5 = run_dir / 'query-feats-aliked-n16.h5'
        matches_h5 = run_dir / f'matches-query-netvlad{self.runtime.k_max}.h5'
        timings_ms: dict[str, float] = {}

        t = time.perf_counter()
        self.extract_features_to_h5(self.netvlad_model, self.netvlad_conf, query_image, query_name, query_global_h5, as_half=True)
        timings_ms['extract_netvlad'] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        self.extract_features_to_h5(self.local_model, self.local_conf, query_image, query_name, query_local_h5, as_half=True)
        timings_ms['extract_local'] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        all_candidates = self.retrieve_topk(query_global_h5, query_name, self.runtime.k_max)
        timings_ms['retrieval'] = (time.perf_counter() - t) * 1000.0
        if not all_candidates:
            return {'success': False, 'failure_reason': 'no_retrieval_candidates', 'timings_ms': timings_ms}

        t = time.perf_counter()
        self.match_pairs_to_h5(query_name, all_candidates, query_local_h5, self.db_local_feats, matches_h5)
        timings_ms['match_local'] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        match_count_by_name = {}
        for db_name in all_candidates:
            try:
                ms, _ = get_matches(matches_h5, query_name, db_name)
                match_count_by_name[db_name] = int(len(ms))
            except Exception:
                match_count_by_name[db_name] = 0

        selected_k = None
        pos_xy = None
        pos_info = None
        k_values = list(range(self.runtime.k_init, self.runtime.k_max + 1, self.runtime.k_step))
        if self.runtime.k_max not in k_values:
            k_values.append(self.runtime.k_max)
        for k in k_values:
            cands = all_candidates[: min(k, len(all_candidates))]
            pos_xy, pos_info = compute_weighted_topometric_position(
                candidates=cands,
                match_count=match_count_by_name,
                ref_xy=self.ref_xy,
                min_match_high=self.runtime.min_match_high,
                min_match_fallback=self.runtime.min_match_fallback,
            )
            if pos_xy is not None:
                selected_k = k
                break
        timings_ms['position'] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        qcam, q_w, q_h = get_query_camera(query_image)
        ranked_pnp_candidates = sorted(
            all_candidates,
            key=lambda name: (-match_count_by_name.get(name, 0), all_candidates.index(name)),
        )
        pnp_candidate_k = min(len(all_candidates), max(self.runtime.k_init, selected_k or 0, self.runtime.k_max))
        pnp_candidates = ranked_pnp_candidates[:pnp_candidate_k]
        heading_deg, pnp_info = estimate_heading_with_pnp(
            reconstruction=self.reconstruction,
            query_name=query_name,
            query_feature_path=query_local_h5,
            matches_path=matches_h5,
            db_names=pnp_candidates,
            query_camera=qcam,
            transform_2x3=self.T_2x3,
            ransac_thresh=self.runtime.ransac_thresh,
        )
        timings_ms['pnp'] = (time.perf_counter() - t) * 1000.0
        timings_ms['total_inner'] = sum(timings_ms.values())

        pnp_inliers = int(pnp_info.get('num_inliers', 0))
        pnp_ok = heading_deg is not None and pnp_inliers >= self.runtime.min_pnp_inliers
        best_name = max(match_count_by_name.items(), key=lambda kv: kv[1])[0] if match_count_by_name else None
        best_m = int(match_count_by_name.get(best_name, 0)) if best_name else 0

        # Position intentionally comes from weighted/topometric candidates; PnP is heading/diagnostics only.
        success = pos_xy is not None
        reason = None if success else 'insufficient_matches_after_k_expansion'

        pnp_floor_xy = pnp_info.get('floorplan_xy')
        consistency_px = None
        if pos_xy is not None and pnp_floor_xy is not None:
            consistency_px = math.hypot(float(pos_xy[0]) - float(pnp_floor_xy[0]), float(pos_xy[1]) - float(pnp_floor_xy[1]))

        final_xy = pos_xy
        candidate_dispersion_px = compute_candidate_dispersion_px(
            pos_info.get('used_candidates') if pos_info else [],
            pos_info.get('weights') if pos_info else {},
            self.ref_xy,
            pos_xy,
        )

        confidence = min(1.0, (best_m / 150.0))
        if pos_info and pos_info.get('strategy') == 'weighted_spatial_cluster':
            confidence = max(confidence, 0.6)
            if candidate_dispersion_px is not None:
                if candidate_dispersion_px > 220:
                    confidence *= 0.55
                elif candidate_dispersion_px > 160:
                    confidence *= 0.7
                elif candidate_dispersion_px > 100:
                    confidence *= 0.85
        elif pos_info and pos_info.get('strategy') == 'best_single_fallback':
            confidence = min(confidence, 0.55)
        if not success:
            confidence = 0.0

        candidate_floorplan_xy = {
            n: [float(self.ref_xy[n][0]), float(self.ref_xy[n][1])]
            for n in match_count_by_name
            if n in self.ref_xy
        }

        self.last_candidates = all_candidates[:10]
        return {
            'success': bool(success),
            'x': float(final_xy[0]) if final_xy is not None else None,
            'y': float(final_xy[1]) if final_xy is not None else None,
            'heading_deg': float(heading_deg) if pnp_ok and heading_deg is not None else None,
            'confidence': float(confidence),
            'selected_k': int(selected_k) if selected_k is not None else None,
            'num_matches_best': int(best_m),
            'best_candidate': best_name,
            'inliers_pnp': int(pnp_inliers),
            'failure_reason': reason,
            'query': {'image_name': query_name, 'width': int(q_w), 'height': int(q_h)},
            'position_debug': {
                'k_schedule': k_values,
                'strategy': pos_info.get('strategy') if pos_info else None,
                'cluster_center': pos_info.get('cluster_center') if pos_info else None,
                'cluster_radius_px': pos_info.get('cluster_radius_px') if pos_info else None,
                'cluster_score': pos_info.get('cluster_score') if pos_info else None,
                'used_candidates': pos_info.get('used_candidates') if pos_info else [],
                'weights': pos_info.get('weights') if pos_info else {},
                'weighted_xy': [float(pos_xy[0]), float(pos_xy[1])] if pos_xy is not None else None,
                'pnp_floor_xy': pnp_floor_xy,
                'weighted_pnp_consistency_px': float(consistency_px) if consistency_px is not None else None,
                'candidate_dispersion_px': float(candidate_dispersion_px) if candidate_dispersion_px is not None else None,
            },
            'pnp_debug': pnp_info,
            'match_count_by_candidate': {k: int(v) for k, v in match_count_by_name.items()},
            'candidate_floorplan_xy': candidate_floorplan_xy,
            'timings_ms': {k: round(v, 3) for k, v in timings_ms.items()},
            'engine': {
                'mode': 'in_memory',
                'k_init': self.runtime.k_init,
                'k_max': self.runtime.k_max,
                'k_step': self.runtime.k_step,
                'query_resize_max': self.query_resize_max,
                'local_feature': 'aliked-n16',
                'matcher': 'lightglue',
            },
        }
