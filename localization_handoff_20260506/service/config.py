from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:  # Support both `python -m service.server` and direct `python service/server.py`.
    from .schemas import PackageConfig, ResolvedPaths, RuntimeSettings
except ImportError:  # pragma: no cover - exercised by direct script launch.
    from schemas import PackageConfig, ResolvedPaths, RuntimeSettings

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / 'config' / 'package_manifest.json'


def _as_path(root: Path, value: str, field: str) -> Path:
    if not value:
        raise ValueError(f'manifest path is empty: {field}')
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def _read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return data


def load_config(manifest_path: str | Path | None = None) -> PackageConfig:
    """Load package manifest and resolve package-contained runtime paths."""

    manifest_file = Path(manifest_path).expanduser() if manifest_path else DEFAULT_MANIFEST_PATH
    if not manifest_file.is_absolute():
        manifest_file = (Path.cwd() / manifest_file).resolve()
    manifest = _read_json(manifest_file)
    root = manifest_file.parent.parent.resolve()

    data = manifest.get('data', {})
    reference_set = data.get('reference_set', {})
    baseline = data.get('baseline', {})
    floorplan = data.get('floorplan', {})
    runtime = manifest.get('runtime', {})

    reference_set_dir = _as_path(root, reference_set.get('root', ''), 'data.reference_set.root')
    reference_images_dir = _as_path(root, reference_set.get('reference_images', ''), 'data.reference_set.reference_images')
    reference_db_dir = _as_path(root, reference_set.get('reference_db', ''), 'data.reference_set.reference_db')
    topometric_dir = _as_path(root, reference_set.get('topometric', ''), 'data.reference_set.topometric')
    baseline_dir = _as_path(root, baseline.get('root', ''), 'data.baseline.root')

    work_dir = _as_path(root, runtime.get('work_dir', 'runtime/localization_runs'), 'runtime.work_dir')
    tmp_dir = _as_path(root, runtime.get('tmp_dir', 'runtime/tmp_queries'), 'runtime.tmp_dir')
    static_dir = _as_path(root, runtime.get('static_dir', 'static'), 'runtime.static_dir')

    paths = ResolvedPaths(
        package_root=root,
        reference_set_dir=reference_set_dir,
        reference_images_dir=reference_images_dir,
        reference_db_dir=reference_db_dir,
        db_global_feats=_as_path(root, reference_set.get('db_global_feats', ''), 'data.reference_set.db_global_feats'),
        db_local_feats=_as_path(root, reference_set.get('db_local_feats', ''), 'data.reference_set.db_local_feats'),
        topometric_dir=topometric_dir,
        ref_points_2d=_as_path(root, reference_set.get('ref_points_2d', ''), 'data.reference_set.ref_points_2d'),
        transform_json=_as_path(root, reference_set.get('transform_json', ''), 'data.reference_set.transform_json'),
        baseline_dir=baseline_dir,
        sfm_model_dir=_as_path(root, baseline.get('triangulated_model', ''), 'data.baseline.triangulated_model'),
        floorplan_path=_as_path(root, floorplan.get('path', ''), 'data.floorplan.path'),
        work_dir=work_dir,
        tmp_dir=tmp_dir,
        static_dir=static_dir,
    )

    settings = RuntimeSettings(
        host=str(runtime.get('host', '0.0.0.0')),
        port=int(runtime.get('port', 8034)),
        max_post_bytes=int(runtime.get('max_post_bytes', 25 * 1024 * 1024)),
        k_init=int(runtime.get('k_init', 20)),
        k_max=int(runtime.get('k_max', 60)),
        k_step=int(runtime.get('k_step', 20)),
        min_match_high=int(runtime.get('min_match_high', 75)),
        min_match_fallback=int(runtime.get('min_match_fallback', 30)),
        ransac_thresh=float(runtime.get('ransac_thresh', 12.0)),
        min_pnp_inliers=int(runtime.get('min_pnp_inliers', 20)),
        query_resize_max=int(runtime.get('query_resize_max', 1280)),
    )
    return PackageConfig(manifest=manifest, paths=paths, runtime=settings)


def load_transform_json(config: PackageConfig | None = None) -> dict[str, Any] | None:
    cfg = config or load_config()
    if not cfg.paths.transform_json.exists():
        return None
    return _read_json(cfg.paths.transform_json)


def resolve_floorplan_path(config: PackageConfig | None = None) -> Path | None:
    """Resolve the package-contained floorplan path.

    The manifest is authoritative. transform.json is still read by the localizer for
    the 2x3 matrix, but it also stores a relative `sources.floorplan` for portability.
    """

    cfg = config or load_config()
    manifest_floorplan = cfg.paths.floorplan_path
    if manifest_floorplan.exists():
        return manifest_floorplan

    transform = load_transform_json(cfg)
    if not transform:
        return manifest_floorplan
    raw = transform.get('sources', {}).get('floorplan')
    if not raw:
        return manifest_floorplan
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    candidates = [
        cfg.paths.transform_json.parent / p,
        cfg.paths.package_root / p,
        cfg.paths.reference_set_dir / p,
    ]
    return next((c for c in candidates if c.exists()), candidates[0])


def required_paths(config: PackageConfig | None = None) -> dict[str, Path]:
    cfg = config or load_config()
    return cfg.paths.required_path_map()


def status_for_paths(config: PackageConfig | None = None) -> dict[str, dict[str, object]]:
    cfg = config or load_config()
    return {k: {'path': str(v), 'exists': v.exists()} for k, v in required_paths(cfg).items()}
