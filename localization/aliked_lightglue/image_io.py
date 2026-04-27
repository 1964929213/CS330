from __future__ import annotations

from pathlib import Path

import numpy as np


def is_dng(path: str | Path) -> bool:
    return str(path).lower().endswith('.dng')


def read_image(path: str | Path, grayscale: bool = False):
    path = Path(path)
    if is_dng(path):
        import pycolmap
        bitmap = pycolmap.Bitmap.read(str(path), not grayscale)
        if bitmap is None or getattr(bitmap, 'is_empty', False):
            raise ValueError(f'无法读取图像: {path}')
        image = bitmap.to_array()
        if grayscale and image.ndim == 3:
            image = np.round(image[..., :3].mean(axis=2)).astype(np.uint8)
        return image

    from hloc.utils.io import read_image as hloc_read_image
    return hloc_read_image(path, grayscale=grayscale)


def image_size(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    if is_dng(path):
        import pycolmap
        cam = pycolmap.infer_camera_from_image(path)
        return int(cam.width), int(cam.height)

    from PIL import Image
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def query_camera(path: str | Path):
    import pycolmap
    w, h = image_size(path)
    f = float(max(w, h))
    cam = pycolmap.Camera(
        model='SIMPLE_RADIAL',
        width=w,
        height=h,
        params=np.array([f, w / 2.0, h / 2.0, 0.0]),
    )
    return cam, w, h
