from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Dict


def _read_next_bytes(fid, num_bytes: int, fmt: str, endian: str = '<'):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


def _has_model(path: Path, ext: str) -> bool:
    return all((path / f'{name}{ext}').is_file() for name in ['cameras', 'images', 'points3D'])


def _read_images_text(path: Path) -> Dict[int, dict]:
    images: Dict[int, dict] = {}
    with path.open('r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elems = line.split()
            image_id = int(elems[0])
            images[image_id] = {
                'image_id': image_id,
                'name': elems[9],
                'camera_id': int(elems[8]),
                'qvec': tuple(float(x) for x in elems[1:5]),
                'tvec': tuple(float(x) for x in elems[5:8]),
            }
            f.readline()
    return images


def _read_images_binary(path: Path) -> Dict[int, dict]:
    images: Dict[int, dict] = {}
    with path.open('rb') as f:
        count = _read_next_bytes(f, 8, 'Q')[0]
        for _ in range(count):
            vals = _read_next_bytes(f, 64, 'idddddddi')
            image_id = int(vals[0])
            name_bytes = []
            c = _read_next_bytes(f, 1, 'c')[0]
            while c != b'\x00':
                name_bytes.append(c)
                c = _read_next_bytes(f, 1, 'c')[0]
            n_points = _read_next_bytes(f, 8, 'Q')[0]
            _read_next_bytes(f, 24 * n_points, 'ddq' * n_points)
            images[image_id] = {
                'image_id': image_id,
                'name': b''.join(name_bytes).decode('utf-8'),
                'camera_id': int(vals[8]),
                'qvec': tuple(float(x) for x in vals[1:5]),
                'tvec': tuple(float(x) for x in vals[5:8]),
            }
    return images


def read_colmap_images(model_dir: str | Path) -> Dict[int, dict]:
    model_dir = Path(model_dir)
    if _has_model(model_dir, '.bin'):
        return _read_images_binary(model_dir / 'images.bin')
    if _has_model(model_dir, '.txt'):
        return _read_images_text(model_dir / 'images.txt')
    raise FileNotFoundError(f'没有找到完整 COLMAP 模型: {model_dir}')


def write_image_pose_table(images: Dict[int, dict], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_name', 'camera_id', 'qvec_w', 'qvec_x', 'qvec_y', 'qvec_z', 'tvec_x', 'tvec_y', 'tvec_z'])
        for image_id in sorted(images):
            item = images[image_id]
            writer.writerow([item['image_id'], item['name'], item['camera_id'], *item['qvec'], *item['tvec']])
