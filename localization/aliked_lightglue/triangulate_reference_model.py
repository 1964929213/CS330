#!/usr/bin/env python3
"""
中间步骤：把 ALIKED + LightGlue 的参考库特征/匹配，triangulate 成适配 hloc 定位的模型。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from localization.aliked_lightglue.colmap_utils import read_colmap_images


@dataclass
class TriangulationConfig:
    reference_model_dir: str
    images_dir: str
    pairs: str
    features: str
    matches: str
    output_dir: str
    overwrite: bool
    verbose: bool
    estimate_two_view_geometries: bool
    skip_geometric_verification: bool
    min_match_score: float | None


def ensure_hloc_import():
    try:
        from hloc import triangulation  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "未检测到 hloc triangulation 依赖。请先安装官方仓库后重试：\n"
            "  git clone --recursive https://github.com/cvg/Hierarchical-Localization.git\n"
            "  cd Hierarchical-Localization\n"
            "  python -m pip install -e .\n"
            f"原始错误: {e}"
        )


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split()
        pairs.append((a, b))
    return pairs


def filter_pairs_to_registered(
    pairs: list[tuple[str, str]],
    registered_names: set[str],
) -> tuple[list[tuple[str, str]], int]:
    kept: list[tuple[str, str]] = []
    dropped = 0
    for a, b in pairs:
        if a in registered_names and b in registered_names:
            kept.append((a, b))
        else:
            dropped += 1
    return kept, dropped


def write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = ''.join(f'{a} {b}\n' for a, b in pairs)
    path.write_text(text, encoding='utf-8')


def triangulate(cfg: TriangulationConfig) -> None:
    ensure_hloc_import()
    from hloc import triangulation

    reference_model_dir = Path(cfg.reference_model_dir).expanduser().resolve()
    images_dir = Path(cfg.images_dir).expanduser().resolve()
    pairs_path = Path(cfg.pairs).expanduser().resolve()
    features_path = Path(cfg.features).expanduser().resolve()
    matches_path = Path(cfg.matches).expanduser().resolve()
    output_dir = Path(cfg.output_dir).expanduser().resolve()

    for p, name in [
        (reference_model_dir, 'reference_model_dir'),
        (images_dir, 'images_dir'),
        (pairs_path, 'pairs'),
        (features_path, 'features'),
        (matches_path, 'matches'),
    ]:
        if not p.exists():
            raise FileNotFoundError(f'{name} 不存在: {p}')

    if cfg.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / 'run_logs'
    run_dir.mkdir(parents=True, exist_ok=True)

    registered = set(item['name'] for item in read_colmap_images(reference_model_dir).values())
    raw_pairs = read_pairs(pairs_path)
    kept_pairs, dropped_pairs = filter_pairs_to_registered(raw_pairs, registered)
    if not kept_pairs:
        raise ValueError('过滤到已注册图像后，没有可用的 pairs')

    filtered_pairs_path = output_dir / f'{pairs_path.stem}-registered{pairs_path.suffix}'
    write_pairs(filtered_pairs_path, kept_pairs)

    mapper_options = {}
    reconstruction = triangulation.main(
        sfm_dir=output_dir,
        reference_model=reference_model_dir,
        image_dir=images_dir,
        pairs=filtered_pairs_path,
        features=features_path,
        matches=matches_path,
        skip_geometric_verification=cfg.skip_geometric_verification,
        estimate_two_view_geometries=cfg.estimate_two_view_geometries,
        min_match_score=cfg.min_match_score,
        verbose=cfg.verbose,
        mapper_options=mapper_options,
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        'timestamp': ts,
        'config': asdict(cfg),
        'artifacts': {
            'filtered_pairs': str(filtered_pairs_path),
            'database': str(output_dir / 'database.db'),
            'triangulated_model_dir': str(output_dir),
        },
        'stats': {
            'registered_images': len(registered),
            'pairs_input': len(raw_pairs),
            'pairs_kept': len(kept_pairs),
            'pairs_dropped': dropped_pairs,
            'reconstruction_summary': reconstruction.summary(),
        },
    }
    summary_path = run_dir / f'triangulation_summary_{ts}.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n[DONE]  triangulation 完成（ALIKED + LightGlue）')
    print(f'- filtered_pairs: {filtered_pairs_path}')
    print(f'- triangulated_model_dir: {output_dir}')
    print(f'- summary: {summary_path}')
    print(reconstruction.summary())



def parse_args(argv: Iterable[str]) -> TriangulationConfig:
    parser = argparse.ArgumentParser(
        description='参考模型 triangulation（ALIKED + LightGlue -> triangulated model）'
    )
    parser.add_argument('--reference-model-dir', required=True, help='原始参考模型目录（作为 triangulation 的 reference_model）')
    parser.add_argument('--images-dir', required=True, help='参考图像目录')
    parser.add_argument('--pairs', required=True, help='步骤1输出的 pairs 文件')
    parser.add_argument('--features', required=True, help='步骤1输出的 ALIKED 特征 h5')
    parser.add_argument('--matches', required=True, help='步骤1输出的 LightGlue 匹配 h5')
    parser.add_argument('--output-dir', required=True, help='triangulated model 输出目录')
    parser.add_argument('--overwrite', action='store_true', help='覆盖 output-dir')
    parser.add_argument('--verbose', action='store_true', help='打印更多 hloc/pycolmap 日志')
    parser.add_argument('--estimate-two-view-geometries', action='store_true', help='使用 hloc 的 two-view estimation 路径')
    parser.add_argument('--skip-geometric-verification', action='store_true', help='跳过 geometric verification')
    parser.add_argument('--min-match-score', type=float, default=None, help='可选的最小匹配分数过滤')

    args = parser.parse_args(list(argv))
    return TriangulationConfig(
        reference_model_dir=args.reference_model_dir,
        images_dir=args.images_dir,
        pairs=args.pairs,
        features=args.features,
        matches=args.matches,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
        estimate_two_view_geometries=bool(args.estimate_two_view_geometries),
        skip_geometric_verification=bool(args.skip_geometric_verification),
        min_match_score=args.min_match_score,
    )


if __name__ == '__main__':
    try:
        cfg = parse_args(sys.argv[1:])
        triangulate(cfg)
    except Exception as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        raise SystemExit(1)
