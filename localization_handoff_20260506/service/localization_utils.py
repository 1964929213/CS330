from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_reference_points(path: Path) -> dict[str, tuple[float, float]]:
    """Read package topometric reference image positions.

    The CSV image names are the stable join key shared by reference images,
    NetVLAD H5 keys, ALIKED H5 keys, and localization results.
    """

    out: dict[str, tuple[float, float]] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_name', 'u_floorplan', 'v_floorplan'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'reference_points_2d.csv missing columns: {sorted(missing)}')
        for row in reader:
            out[row['image_name']] = (float(row['u_floorplan']), float(row['v_floorplan']))
    if not out:
        raise ValueError('reference_points_2d.csv is empty')
    return out


def load_transform_2x3(transform_path: Path) -> np.ndarray | None:
    """Load the topometric 2x3 world-to-floorplan transform if available."""

    if not transform_path.exists():
        return None
    data = json.loads(transform_path.read_text(encoding='utf-8'))
    raw = data.get('matrix_2x3', data.get('T', []))
    transform = np.array(raw, dtype=np.float64)
    if transform.shape != (2, 3):
        return None
    return transform


def get_query_camera(query_image: Path):
    """Create a simple query camera from the uploaded image dimensions."""

    import pycolmap

    with Image.open(query_image) as im:
        width, height = im.size

    focal = float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    camera = pycolmap.Camera(
        model='SIMPLE_RADIAL',
        width=int(width),
        height=int(height),
        params=np.array([focal, cx, cy, 0.0]),
    )
    return camera, width, height


def compute_weighted_topometric_position(
    candidates: list[str],
    match_count: dict[str, int],
    ref_xy: dict[str, tuple[float, float]],
    min_match_high: int,
    min_match_fallback: int,
    cluster_radius_px: float = 180.0,
):
    """Compute the final floorplan position from weighted topometric candidates.

    This preserves the handoff contract: final x/y is not derived from PnP.
    """

    valid = [(name, match_count.get(name, 0)) for name in candidates if name in ref_xy]
    high = [(name, matches) for name, matches in valid if matches > min_match_high]

    info: dict[str, Any] = {
        'strategy': None,
        'used_candidates': [],
        'weights': {},
        'cluster_radius_px': float(cluster_radius_px),
    }

    def dist_to(name: str, center: str) -> float:
        x0, y0 = ref_xy[center]
        x1, y1 = ref_xy[name]
        return math.hypot(float(x1) - float(x0), float(y1) - float(y0))

    def weighted(items: list[tuple[str, int]], strategy: str):
        total = float(sum(matches for _, matches in items))
        x = 0.0
        y = 0.0
        for name, matches in items:
            weight = float(matches) / total
            px, py = ref_xy[name]
            x += weight * px
            y += weight * py
            info['weights'][name] = weight
            info['used_candidates'].append(name)
        info['strategy'] = strategy
        return (x, y), info

    if high:
        rank = {name: i for i, name in enumerate(candidates)}
        best_cluster = None
        for center, _matches in high:
            cluster = [
                (name, matches)
                for name, matches in valid
                if matches > min_match_fallback and dist_to(name, center) <= cluster_radius_px
            ]
            if not cluster:
                continue
            score = sum(float(matches) / math.sqrt(rank.get(name, 9999) + 1.0) for name, matches in cluster)
            tie = -rank.get(center, 9999)
            key = (score, len(cluster), tie)
            if best_cluster is None or key > best_cluster[0]:
                best_cluster = (key, center, cluster, score)
        if best_cluster is not None:
            _key, center, cluster, score = best_cluster
            info['cluster_center'] = center
            info['cluster_score'] = float(score)
            return weighted(cluster, 'weighted_spatial_cluster')

    fallback = [(name, matches) for name, matches in valid if matches > min_match_fallback]
    if fallback:
        best = max(fallback, key=lambda item: item[1])[0]
        info['strategy'] = 'best_single_fallback'
        info['used_candidates'] = [best]
        info['weights'][best] = 1.0
        info['cluster_center'] = best
        return ref_xy[best], info

    info['strategy'] = 'failed'
    return None, info


def estimate_heading_with_pnp(
    reconstruction,
    query_name: str,
    query_feature_path: Path,
    matches_path: Path,
    db_names: list[str],
    query_camera,
    transform_2x3: np.ndarray | None,
    ransac_thresh: float,
):
    """Estimate PnP heading and diagnostics while leaving final x/y topometric."""

    import pycolmap  # noqa: F401 - required by pycolmap pose estimation bindings
    from hloc.utils.io import get_keypoints, get_matches

    db_name_to_id = {image.name: image_id for image_id, image in reconstruction.images.items()}
    query_keypoints = get_keypoints(query_feature_path, query_name) + 0.5

    kp_to_3d: dict[int, set[int]] = {}
    total_pair_matches = 0
    used_db = []

    for db_name in db_names:
        db_id = db_name_to_id.get(db_name)
        if db_id is None:
            continue
        image = reconstruction.images[db_id]
        point3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D])

        try:
            matches, _scores = get_matches(matches_path, query_name, db_name)
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
            if point3d_id < 0:
                continue
            kp_to_3d.setdefault(int(query_idx), set()).add(point3d_id)

    matched_keypoint_indices = []
    matched_point3d_ids = []
    for query_idx, point3d_ids in kp_to_3d.items():
        for point3d_id in point3d_ids:
            matched_keypoint_indices.append(query_idx)
            matched_point3d_ids.append(point3d_id)

    if not matched_keypoint_indices:
        return None, {
            'num_corr': 0,
            'num_pair_matches': total_pair_matches,
            'used_db_for_pnp': used_db,
            'failure': 'no_2d3d_correspondences',
        }

    points2d = query_keypoints[np.array(matched_keypoint_indices)]
    points3d = [reconstruction.points3D[int(point3d_id)].xyz for point3d_id in matched_point3d_ids]

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

    rotation = np.array(ret['cam_from_world'].rotation.matrix())
    translation = np.array(ret['cam_from_world'].translation, dtype=np.float64).reshape(3)

    forward_world = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dx = float(forward_world[0])
    dz = float(forward_world[2])

    camera_center = -rotation.T @ translation.reshape(3, 1)
    cx = float(camera_center[0, 0])
    cy = float(camera_center[1, 0])
    cz = float(camera_center[2, 0])

    floorplan_xy = None
    if transform_2x3 is not None:
        linear = transform_2x3[:, :2]
        du, dv = linear @ np.array([dx, dz], dtype=np.float64)
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
        'pnp': {
            'qvec_xyzw': [
                float(ret['cam_from_world'].rotation.quat[0]),
                float(ret['cam_from_world'].rotation.quat[1]),
                float(ret['cam_from_world'].rotation.quat[2]),
                float(ret['cam_from_world'].rotation.quat[3]),
            ],
            'tvec': [float(x) for x in np.array(ret['cam_from_world'].translation).reshape(-1).tolist()],
        },
    }
