#!/usr/bin/env python3
"""
UNav 步骤1：参考图库与定位索引构建（NetVLAD + ALIKED + LightGlue）。

当前配置：NetVLAD 检索，ALIKED 特征，LightGlue 匹配。
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

from localization.aliked_lightglue.colmap_utils import read_colmap_images, write_image_pose_table
from localization.aliked_lightglue.feature_pipeline import extract_features as run_extract_features, match_features as run_match_features

IMAGE_GLOBS = [
    '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.dng',
    '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP', '*.DNG',
]


@dataclass
class BuildConfig:
    images_dir: str
    sfm_model_dir: str
    output_dir: str
    num_matched: int
    local_feature_conf: str
    matcher_conf: str
    overwrite: bool
    export_half: bool


def ensure_hloc_import():
    try:
        from hloc import extract_features as hloc_extract, match_features as hloc_match, pairs_from_retrieval  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "未检测到 hloc。请先安装官方仓库后重试：\n"
            "  git clone --recursive https://github.com/cvg/Hierarchical-Localization.git\n"
            "  cd Hierarchical-Localization\n"
            "  python -m pip install -e .\n"
            f"原始错误: {e}"
        )


def collect_image_list(images_dir: Path) -> list[str]:
    names: set[str] = set()
    for pat in IMAGE_GLOBS:
        for p in images_dir.rglob(pat):
            if p.is_file():
                names.add(p.relative_to(images_dir).as_posix())
    return sorted(names)


def write_reference_points_template(images: dict[int, dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['image_name', 'u_floorplan', 'v_floorplan'])
        for iid in sorted(images.keys()):
            item = images[iid]
            writer.writerow([item['name'], '', ''])


def build(cfg: BuildConfig) -> None:
    ensure_hloc_import()
    from hloc import extract_features as hloc_extract, match_features as hloc_match, pairs_from_retrieval

    images_dir = Path(cfg.images_dir).expanduser().resolve()
    sfm_model_dir = Path(cfg.sfm_model_dir).expanduser().resolve()
    output_dir = Path(cfg.output_dir).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir 不存在: {images_dir}")
    if not sfm_model_dir.exists():
        raise FileNotFoundError(f"sfm_model_dir 不存在: {sfm_model_dir}")

    if cfg.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / "run_logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    images = read_colmap_images(sfm_model_dir)
    if len(images) == 0:
        raise RuntimeError("COLMAP 模型中未读取到任何已注册图像")
    pose_table = output_dir / "image_pose_table.csv"
    write_image_pose_table(images, pose_table)

    topometric_dir = output_dir / 'topometric'
    topometric_dir.mkdir(parents=True, exist_ok=True)
    ref_points_template = topometric_dir / 'reference_points_2d_template.csv'
    write_reference_points_template(images, ref_points_template)

    retrieval_conf = hloc_extract.confs["netvlad"]
    image_list = collect_image_list(images_dir)
    if not image_list:
        raise ValueError(f"在参考图像目录下未找到支持的图片: {images_dir}")
    retrieval_path = run_extract_features(
        retrieval_conf,
        images_dir,
        image_list=image_list,
        feature_path=output_dir / f"{retrieval_conf['output']}.h5",
        as_half=cfg.export_half,
        overwrite=cfg.overwrite,
    )

    if cfg.local_feature_conf not in hloc_extract.confs:
        raise ValueError(
            f"local_feature_conf={cfg.local_feature_conf} 不在 hloc 标准配置中，可选: "
            f"{sorted(hloc_extract.confs.keys())}"
        )
    feature_conf = hloc_extract.confs[cfg.local_feature_conf]
    local_feature_path = run_extract_features(
        feature_conf,
        images_dir,
        image_list=image_list,
        feature_path=output_dir / f"{feature_conf['output']}.h5",
        as_half=cfg.export_half,
        overwrite=cfg.overwrite,
    )

    pairs_path = output_dir / f"pairs-db-netvlad{cfg.num_matched}.txt"
    pairs_from_retrieval.main(
        descriptors=retrieval_path,
        output=pairs_path,
        num_matched=cfg.num_matched,
        db_model=sfm_model_dir,
    )

    if cfg.matcher_conf not in hloc_match.confs:
        raise ValueError(
            f"matcher_conf={cfg.matcher_conf} 不在 hloc 标准配置中，可选: "
            f"{sorted(hloc_match.confs.keys())}"
        )
    matcher_conf = hloc_match.confs[cfg.matcher_conf]
    match_path = run_match_features(
        matcher_conf,
        pairs_path,
        match_path=output_dir / f"{feature_conf['output']}_{matcher_conf['output']}_{pairs_path.stem}.h5",
        feature_path_q=local_feature_path,
        feature_path_ref=local_feature_path,
        overwrite=cfg.overwrite,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": ts,
        "config": asdict(cfg),
        "artifacts": {
            "image_pose_table": str(pose_table),
            "reference_points_2d_template": str(ref_points_template),
            "global_descriptors_netvlad": str(retrieval_path),
            "local_features": str(local_feature_path),
            "retrieval_pairs": str(pairs_path),
            "local_matches": str(match_path),
        },
        "stats": {
            "registered_images": len(images),
        },
    }
    summary_path = run_dir / f"build_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE] 参考图库与索引构建完成（NetVLAD + ALIKED + LightGlue）")
    print('[INFO] topometric 标注模板已生成，请在 floorplan 对齐后产出 reference_points_2d.csv 再用于 query。')
    for k, v in summary["artifacts"].items():
        print(f"- {k}: {v}")
    print(f"- summary: {summary_path}")



def parse_args(argv: Iterable[str]) -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="UNav 步骤1：参考图库与索引构建（NetVLAD + ALIKED + LightGlue）"
    )
    parser.add_argument("--images-dir", required=True, help="参考图像目录")
    parser.add_argument("--sfm-model-dir", required=True, help="COLMAP 稀疏模型目录（含 cameras/images/points3D）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--num-matched", type=int, default=20, help="NetVLAD 检索 Top-K（用于生成 pairs）")
    parser.add_argument(
        "--local-feature-conf",
        default="aliked-n16",
        help="hloc.extract_features 配置名，当前默认使用 aliked-n16",
    )
    parser.add_argument(
        "--matcher-conf",
        default="aliked+lightglue",
        help="hloc.match_features 配置名，当前默认使用 aliked+lightglue",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖 output-dir 旧产物")
    parser.add_argument("--no-half", action="store_true", help="禁用 fp16 存储（默认启用）")

    args = parser.parse_args(list(argv))
    local_feature_conf = args.local_feature_conf or "aliked-n16"

    return BuildConfig(
        images_dir=args.images_dir,
        sfm_model_dir=args.sfm_model_dir,
        output_dir=args.output_dir,
        num_matched=args.num_matched,
        local_feature_conf=local_feature_conf,
        matcher_conf=args.matcher_conf,
        overwrite=bool(args.overwrite),
        export_half=not bool(args.no_half),
    )


if __name__ == "__main__":
    try:
        cfg = parse_args(sys.argv[1:])
        build(cfg)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
