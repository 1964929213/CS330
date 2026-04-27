from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def ensure_deps():
    try:
        import h5py  # noqa: F401
        import pycolmap  # noqa: F401
        from hloc import extract_features, match_features, pairs_from_retrieval  # noqa: F401
        from hloc.utils.io import get_keypoints, get_matches  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "缺少依赖。请先安装 hloc、pycolmap、h5py、pillow。\n"
            "hloc: https://github.com/cvg/Hierarchical-Localization\n"
            f"原始错误: {e}"
        )


@dataclass
class Config:
    query_image: str
    sfm_model_dir: str
    db_global_feats: str
    db_local_feats: str
    ref_points_2d: str
    output_json: str
    work_dir: str
    k_init: int
    k_max: int
    k_step: int
    query_resize_max: int
    local_feature_conf: str
    matcher_conf: str
    min_match_high: int
    min_match_low: int
    ransac_thresh: float
    min_pnp_inliers: int
    overwrite: bool


def read_ref_points_2d(path: Path) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_name', 'u_floorplan', 'v_floorplan'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'reference_points_2d.csv 缺少列: {sorted(missing)}')
        for row in reader:
            out[row['image_name']] = (float(row['u_floorplan']), float(row['v_floorplan']))
    if not out:
        raise ValueError('reference_points_2d.csv 为空')
    return out


def compute_weighted_position(
    candidates: List[str],
    match_count: Dict[str, int],
    ref_xy: Dict[str, Tuple[float, float]],
    min_match_high: int,
    min_match_low: int,
):
    valid = [(name, match_count.get(name, 0)) for name in candidates if name in ref_xy]
    strong = [(name, n_match) for name, n_match in valid if n_match > min_match_high]
    info = {'strategy': None, 'used_candidates': [], 'weights': {}}

    if strong:
        total = float(sum(n_match for _, n_match in strong))
        x, y = 0.0, 0.0
        for name, n_match in strong:
            weight = float(n_match) / total
            px, py = ref_xy[name]
            x += weight * px
            y += weight * py
            info['weights'][name] = weight
            info['used_candidates'].append(name)
        info['strategy'] = 'weighted'
        return (x, y), info

    weak = [(name, n_match) for name, n_match in valid if n_match > min_match_low]
    if weak:
        best = max(weak, key=lambda item: item[1])[0]
        info['strategy'] = 'best_low_threshold'
        info['used_candidates'] = [best]
        info['weights'][best] = 1.0
        return ref_xy[best], info

    info['strategy'] = 'failed'
    return None, info


def estimate_heading_with_pnp(
    reconstruction,
    query_name: str,
    query_feature_path: Path,
    matches_path: Path,
    db_names: List[str],
    query_camera,
    transform_2x3: np.ndarray | None,
    ransac_thresh: float,
):
    import pycolmap
    from hloc.utils.io import get_keypoints, get_matches

    db_name_to_id = {img.name: image_id for image_id, img in reconstruction.images.items()}
    query_keypoints = get_keypoints(query_feature_path, query_name) + 0.5

    keypoint_to_3d: Dict[int, set] = {}
    total_pair_matches = 0
    used_db = []

    for db_name in db_names:
        db_id = db_name_to_id.get(db_name)
        if db_id is None:
            continue
        image = reconstruction.images[db_id]
        point3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D])

        try:
            matches, _ = get_matches(matches_path, query_name, db_name)
        except Exception:
            continue

        keep = point3d_ids[matches[:, 1]] != -1
        matches = matches[keep]
        total_pair_matches += int(len(matches))
        if len(matches) == 0:
            continue

        used_db.append(db_name)
        for query_idx, db_idx in matches:
            point3d_id = int(point3d_ids[db_idx])
            if point3d_id >= 0:
                keypoint_to_3d.setdefault(int(query_idx), set()).add(point3d_id)

    keypoint_ids = []
    point3d_ids = []
    for query_idx, ids in keypoint_to_3d.items():
        for point3d_id in ids:
            keypoint_ids.append(query_idx)
            point3d_ids.append(point3d_id)

    if not keypoint_ids:
        return None, {
            'num_corr': 0,
            'num_pair_matches': total_pair_matches,
            'used_db_for_pnp': used_db,
            'failure': 'no_2d3d_correspondences',
        }

    points2d = query_keypoints[np.array(keypoint_ids)]
    points3d = [reconstruction.points3D[int(pid)].xyz for pid in point3d_ids]
    ret = pycolmap.estimate_and_refine_absolute_pose(
        points2d,
        points3d,
        query_camera,
        estimation_options={'ransac': {'max_error': float(ransac_thresh)}},
    )

    if ret is None:
        return None, {
            'num_corr': int(len(points2d)),
            'num_pair_matches': total_pair_matches,
            'used_db_for_pnp': used_db,
            'failure': 'pnp_failed',
        }

    R = np.array(ret['cam_from_world'].rotation.matrix())
    t = np.array(ret['cam_from_world'].translation, dtype=np.float64).reshape(3)
    fwd_world = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dx, dz = float(fwd_world[0]), float(fwd_world[2])
    center = -R.T @ t.reshape(3, 1)
    cx, cy, cz = float(center[0, 0]), float(center[1, 0]), float(center[2, 0])

    floorplan_xy = None
    if transform_2x3 is not None:
        A = transform_2x3[:, :2]
        du, dv = A @ np.array([dx, dz], dtype=np.float64)
        heading_deg = (math.degrees(math.atan2(-dv, du)) + 360.0) % 360.0
        uv = transform_2x3 @ np.array([cx, cz, 1.0], dtype=np.float64)
        floorplan_xy = [float(uv[0]), float(uv[1])]
    else:
        heading_deg = (math.degrees(math.atan2(dz, dx)) + 360.0) % 360.0

    return heading_deg, {
        'num_corr': int(len(points2d)),
        'num_pair_matches': total_pair_matches,
        'used_db_for_pnp': used_db,
        'num_inliers': int(ret.get('num_inliers', 0)),
        'camera_center_world': [cx, cy, cz],
        'floorplan_xy': floorplan_xy,
    }


def parse_retrieval_pairs(path: Path, query_name: str) -> List[str]:
    out = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q, db = line.split()
            if q == query_name:
                out.append(db)
    return out


def load_transform_2x3_if_exists(ref_points_2d_path: Path) -> np.ndarray | None:
    transform_path = ref_points_2d_path.parent / 'transform.json'
    if not transform_path.exists():
        return None
    data = json.loads(transform_path.read_text(encoding='utf-8'))
    matrix = np.array(data.get('matrix_2x3', []), dtype=np.float64)
    if matrix.shape != (2, 3):
        return None
    return matrix


def build_cfg_from_args(args, query_image: Path, output_json: Path) -> Config:
    return Config(
        query_image=str(query_image),
        sfm_model_dir=args.sfm_model_dir,
        db_global_feats=args.db_global_feats,
        db_local_feats=args.db_local_feats,
        ref_points_2d=args.ref_points_2d,
        output_json=str(output_json),
        work_dir=args.work_dir,
        k_init=int(args.k_init),
        k_max=int(args.k_max),
        k_step=int(args.k_step),
        query_resize_max=int(getattr(args, 'query_resize_max', 1280)),
        local_feature_conf=args.local_feature_conf,
        matcher_conf=args.matcher_conf,
        min_match_high=int(args.min_match_high),
        min_match_low=int(args.min_match_low),
        ransac_thresh=float(args.ransac_thresh),
        min_pnp_inliers=int(args.min_pnp_inliers),
        overwrite=bool(args.overwrite),
    )


def collect_images(query_dir: Path, image_glob: str) -> List[Path]:
    patterns = [p.strip() for p in image_glob.split(',') if p.strip()]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(query_dir.rglob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})
