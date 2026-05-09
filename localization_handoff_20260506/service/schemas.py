from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


@dataclass(frozen=True)
class ResolvedPaths:
    """Absolute runtime paths derived from the package manifest."""

    package_root: Path
    reference_set_dir: Path
    reference_images_dir: Path
    reference_db_dir: Path
    db_global_feats: Path
    db_local_feats: Path
    topometric_dir: Path
    ref_points_2d: Path
    transform_json: Path
    baseline_dir: Path
    sfm_model_dir: Path
    floorplan_path: Path
    work_dir: Path
    tmp_dir: Path
    static_dir: Path

    def required_path_map(self) -> dict[str, Path]:
        return {
            'db_global_feats': self.db_global_feats,
            'db_local_feats': self.db_local_feats,
            'sfm_model_dir': self.sfm_model_dir,
            'ref_points_2d': self.ref_points_2d,
            'transform_json': self.transform_json,
        }


@dataclass(frozen=True)
class RuntimeSettings:
    host: str
    port: int
    max_post_bytes: int
    k_init: int
    k_max: int
    k_step: int
    min_match_high: int
    min_match_fallback: int
    ransac_thresh: float
    min_pnp_inliers: int
    query_resize_max: int


@dataclass(frozen=True)
class PackageConfig:
    manifest: dict[str, Any]
    paths: ResolvedPaths
    runtime: RuntimeSettings
