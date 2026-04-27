#!/usr/bin/env python3
"""
UNav 步骤2：单图/批量在线定位（NetVLAD + ALIKED + LightGlue + 加权位置 + PnP）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

from localization.aliked_lightglue import image_io
from localization.aliked_lightglue.feature_pipeline import extract_features as run_extract_features, match_features as run_match_features
from localization.aliked_lightglue.localization_core import (
    Config,
    build_cfg_from_args as _build_cfg_from_args,
    collect_images,
    compute_weighted_position,
    ensure_deps,
    estimate_heading_with_pnp,
    load_transform_2x3_if_exists,
    parse_retrieval_pairs,
    read_ref_points_2d,
)


def parse_args(argv: Iterable[str]):
    parser = argparse.ArgumentParser(
        description="UNav 步骤2：在线定位（NetVLAD + ALIKED + LightGlue + 加权 + PnP）"
    )
    parser.add_argument("--query-image", default="", help="单张查询图像路径")
    parser.add_argument("--query-dir", default="", help="批量模式：查询图目录（递归）")
    parser.add_argument("--output-dir", default="", help="批量模式：每张图输出json目录")
    parser.add_argument("--image-glob", default="*.jpg,*.jpeg,*.png,*.bmp,*.webp,*.dng,*.JPG,*.JPEG,*.PNG,*.BMP,*.WEBP,*.DNG", help="批量模式图像过滤（逗号分隔）")

    parser.add_argument("--sfm-model-dir", required=True, help="COLMAP 稀疏模型目录")
    parser.add_argument("--db-global-feats", required=True, help="步骤1输出：global-feats-netvlad.h5")
    parser.add_argument("--db-local-feats", required=True, help="步骤1输出：feats-aliked-*.h5")
    parser.add_argument("--ref-points-2d", required=True, help="topometric/reference_points_2d.csv")
    parser.add_argument("--output-json", default="", help="单图模式：定位结果JSON输出路径")
    parser.add_argument("--work-dir", default="./localization_runs_aliked_lightglue", help="中间产物目录")

    parser.add_argument("--k-init", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--query-resize-max", type=int, default=1280, help="仅 query 侧缩图上限，长边超过时会先缩小以提速")

    parser.add_argument("--local-feature-conf", default="aliked-n16")
    parser.add_argument("--matcher-conf", default="aliked+lightglue")

    parser.add_argument("--min-match-high", type=int, default=75, help="论文阈值（高置信匹配）")
    parser.add_argument("--min-match-low", type=int, default=30, help="论文低阈值")
    parser.add_argument("--ransac-thresh", type=float, default=8.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=20)

    parser.add_argument("--overwrite", action="store_true", help="若 work-dir 存在则先删除")

    args = parser.parse_args(list(argv))

    if bool(args.query_image) == bool(args.query_dir):
        raise ValueError("必须二选一：--query-image 或 --query-dir")
    if args.query_image and not args.output_json:
        raise ValueError("单图模式必须提供 --output-json")
    if args.query_dir and not args.output_dir:
        raise ValueError("批量模式必须提供 --output-dir")

    if args.k_init <= 0 or args.k_step <= 0 or args.k_max <= 0:
        raise ValueError("k-init/k-step/k-max 必须为正整数")
    if args.k_init > args.k_max:
        raise ValueError("k-init 不能大于 k-max")

    args.local_feature_conf = args.local_feature_conf or "aliked-n16"
    return args


def build_cfg_from_args(args, query_image: Path, output_json: Path) -> Config:
    base = _build_cfg_from_args(args, query_image=query_image, output_json=output_json)
    base.local_feature_conf = args.local_feature_conf
    base.matcher_conf = args.matcher_conf
    return base


def run_single(cfg: Config):
    ensure_deps()

    import pycolmap
    from hloc import match_features as hloc_match, pairs_from_retrieval
    from hloc.utils.io import get_matches

    query_image = Path(cfg.query_image).expanduser().resolve()
    sfm_model_dir = Path(cfg.sfm_model_dir).expanduser().resolve()
    db_global_feats = Path(cfg.db_global_feats).expanduser().resolve()
    db_local_feats = Path(cfg.db_local_feats).expanduser().resolve()
    ref_points_2d = Path(cfg.ref_points_2d).expanduser().resolve()
    output_json = Path(cfg.output_json).expanduser().resolve()
    work_dir = Path(cfg.work_dir).expanduser().resolve()

    for p, name in [
        (query_image, 'query_image'),
        (sfm_model_dir, 'sfm_model_dir'),
        (db_global_feats, 'db_global_feats'),
        (db_local_feats, 'db_local_feats'),
        (ref_points_2d, 'ref_points_2d'),
    ]:
        if not p.exists():
            raise FileNotFoundError(f'{name} 不存在: {p}')

    work_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = work_dir / f'run_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)

    query_dir = run_dir / 'query'
    query_dir.mkdir(parents=True, exist_ok=True)
    query_name = query_image.name
    query_copy = query_dir / query_name
    shutil.copy2(query_image, query_copy)

    ref_xy = read_ref_points_2d(ref_points_2d)
    T_2x3 = load_transform_2x3_if_exists(ref_points_2d)

    from hloc import extract_features as hloc_extract
    if cfg.local_feature_conf not in hloc_extract.confs:
        raise ValueError(f'local_feature_conf={cfg.local_feature_conf} 不在 hloc 配置中')
    if cfg.matcher_conf not in hloc_match.confs:
        raise ValueError(f'matcher_conf={cfg.matcher_conf} 不在 hloc 配置中')

    query_global_h5 = run_dir / 'query-global-feats-netvlad.h5'
    query_local_h5 = run_dir / f'query-feats-{cfg.local_feature_conf}.h5'

    global_conf_query = json.loads(json.dumps(hloc_extract.confs['netvlad']))
    feature_conf_query = json.loads(json.dumps(hloc_extract.confs[cfg.local_feature_conf]))
    if int(getattr(cfg, 'query_resize_max', 1280)) > 0:
        global_conf_query['preprocessing']['resize_max'] = int(getattr(cfg, 'query_resize_max', 1280))
        feature_conf_query['preprocessing']['resize_max'] = int(getattr(cfg, 'query_resize_max', 1280))

    run_extract_features(
        global_conf_query,
        query_dir,
        image_list=[query_name],
        feature_path=query_global_h5,
        as_half=True,
        overwrite=True,
    )
    run_extract_features(
        feature_conf_query,
        query_dir,
        image_list=[query_name],
        feature_path=query_local_h5,
        as_half=True,
        overwrite=True,
    )

    pairs_kmax = run_dir / f'pairs-query-netvlad{cfg.k_max}.txt'
    pairs_from_retrieval.main(
        descriptors=query_global_h5,
        db_descriptors=db_global_feats,
        output=pairs_kmax,
        num_matched=cfg.k_max,
        db_model=sfm_model_dir,
    )

    all_candidates = parse_retrieval_pairs(pairs_kmax, query_name)
    if not all_candidates:
        out = {
            'success': False,
            'failure_reason': 'no_retrieval_candidates',
            'config': asdict(cfg),
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        return out

    matches_h5 = run_dir / f'matches-{cfg.matcher_conf}-query-netvlad{cfg.k_max}.h5'
    run_match_features(
        hloc_match.confs[cfg.matcher_conf],
        pairs_kmax,
        match_path=matches_h5,
        feature_path_q=query_local_h5,
        feature_path_ref=db_local_feats,
        overwrite=True,
    )

    selected_k = None
    pos_xy = None
    pos_info = None
    match_count_by_name: Dict[str, int] = {}

    for db_name in all_candidates:
        try:
            ms, _ = get_matches(matches_h5, query_name, db_name)
            match_count_by_name[db_name] = int(len(ms))
        except Exception:
            match_count_by_name[db_name] = 0

    k_values = list(range(cfg.k_init, cfg.k_max + 1, cfg.k_step))
    k_values = [k for k in k_values if k > 0]
    if cfg.k_max not in k_values:
        k_values.append(cfg.k_max)

    for k in k_values:
        cands = all_candidates[: min(k, len(all_candidates))]
        pos_xy, pos_info = compute_weighted_position(
            candidates=cands,
            match_count=match_count_by_name,
            ref_xy=ref_xy,
            min_match_high=cfg.min_match_high,
            min_match_low=cfg.min_match_low,
        )
        if pos_xy is not None:
            selected_k = k
            break

    reconstruction = pycolmap.Reconstruction(sfm_model_dir)
    qcam, q_w, q_h = image_io.query_camera(query_image)

    pnp_candidates = all_candidates[: min(selected_k or cfg.k_init, len(all_candidates))]
    heading_deg, pnp_info = estimate_heading_with_pnp(
        reconstruction=reconstruction,
        query_name=query_name,
        query_feature_path=query_local_h5,
        matches_path=matches_h5,
        db_names=pnp_candidates,
        query_camera=qcam,
        transform_2x3=T_2x3,
        ransac_thresh=cfg.ransac_thresh,
    )

    pnp_inliers = int(pnp_info.get('num_inliers', 0))
    pnp_ok = heading_deg is not None and pnp_inliers >= cfg.min_pnp_inliers

    best_name = max(match_count_by_name.items(), key=lambda kv: kv[1])[0] if match_count_by_name else None
    best_m = int(match_count_by_name.get(best_name, 0)) if best_name else 0

    success = (pos_xy is not None) and pnp_ok
    if not success:
        if pos_xy is None:
            reason = 'insufficient_matches_after_k_expansion'
        elif heading_deg is None:
            reason = pnp_info.get('failure', 'pnp_failed')
        else:
            reason = f'pnp_inliers_too_low({pnp_inliers}<{cfg.min_pnp_inliers})'
    else:
        reason = None

    confidence = min(1.0, (best_m / 150.0))
    if pnp_ok:
        confidence = min(1.0, confidence * (0.7 + 0.3 * min(1.0, pnp_inliers / 60.0)))
    else:
        confidence *= 0.5

    out = {
        'success': bool(success),
        'x': float(pos_xy[0]) if pos_xy is not None else None,
        'y': float(pos_xy[1]) if pos_xy is not None else None,
        'heading_deg': float(heading_deg) if heading_deg is not None else None,
        'confidence': float(confidence),
        'num_candidates_total': int(len(all_candidates)),
        'selected_k': int(selected_k) if selected_k is not None else None,
        'num_matches_best': int(best_m),
        'best_candidate': best_name,
        'inliers_pnp': int(pnp_inliers),
        'failure_reason': reason,
        'query': {
            'image_name': query_name,
            'width': int(q_w),
            'height': int(q_h),
        },
        'position_debug': {
            'k_schedule': k_values,
            'strategy': pos_info.get('strategy') if pos_info else None,
            'used_candidates': pos_info.get('used_candidates') if pos_info else [],
            'weights': pos_info.get('weights') if pos_info else {},
        },
        'pnp_debug': pnp_info,
        'match_count_by_candidate': {k: int(v) for k, v in match_count_by_name.items()},
        'artifacts': {
            'run_dir': str(run_dir),
            'query_global_h5': str(query_global_h5),
            'query_local_h5': str(query_local_h5),
            'pairs_kmax': str(pairs_kmax),
            'matches_h5': str(matches_h5),
        },
        'config': asdict(cfg),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


if __name__ == "__main__":
    try:
        args = parse_args(__import__('sys').argv[1:])

        if args.query_image:
            work_dir = Path(args.work_dir).expanduser().resolve()
            if args.overwrite and work_dir.exists():
                shutil.rmtree(work_dir)
            cfg = build_cfg_from_args(
                args,
                query_image=Path(args.query_image).expanduser().resolve(),
                output_json=Path(args.output_json).expanduser().resolve(),
            )
            result = run_single(cfg)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            query_dir = Path(args.query_dir).expanduser().resolve()
            if not query_dir.exists():
                raise FileNotFoundError(f'query-dir 不存在: {query_dir}')
            out_dir = Path(args.output_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            work_dir = Path(args.work_dir).expanduser().resolve()
            if args.overwrite and work_dir.exists():
                shutil.rmtree(work_dir)

            images = collect_images(query_dir, args.image_glob)
            if not images:
                raise ValueError(f'query-dir 下未找到图像: {query_dir}')

            summary = {
                'query_dir': str(query_dir),
                'output_dir': str(out_dir),
                'total': len(images),
                'success': 0,
                'failed': 0,
                'results': [],
            }

            for img in images:
                rel = img.relative_to(query_dir)
                out_json = out_dir / rel.with_suffix('.json')
                out_json.parent.mkdir(parents=True, exist_ok=True)

                cfg = build_cfg_from_args(args, query_image=img, output_json=out_json)
                try:
                    res = run_single(cfg)
                    is_ok = bool(res.get('success', False))
                    summary['success' if is_ok else 'failed'] += 1
                    summary['results'].append({
                        'query_image': str(img),
                        'output_json': str(out_json),
                        'success': is_ok,
                        'failure_reason': res.get('failure_reason'),
                        'inliers_pnp': res.get('inliers_pnp'),
                    })
                    print(f'[OK] {img} -> {out_json} success={is_ok}')
                except Exception as e:
                    summary['failed'] += 1
                    summary['results'].append({
                        'query_image': str(img),
                        'output_json': str(out_json),
                        'success': False,
                        'failure_reason': f'exception: {e}',
                        'inliers_pnp': None,
                    })
                    print(f'[FAIL] {img}: {e}')

            summary_path = out_dir / 'batch_summary.json'
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f'[DONE] batch summary: {summary_path}')

    except Exception as e:
        print(f'[ERROR] {e}')
        raise SystemExit(1)
