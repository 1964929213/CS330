# 重定位

这版先按 0416 的结果整理了一下，后面如果要继续改重定位，可以直接从这里接着做。

现在用的是：

- NetVLAD 检索
- ALIKED 特征
- LightGlue 匹配
- 加权平均算平面图位置
- PnP 算朝向

代码在：

```text
localization/aliked_lightglue/
```

0416 那版已经建好的库在：

```text
localization/baseline_0416_1/
```

里面主要是：

```text
reference_db/
  image_pose_table.csv
  pairs-db-netvlad20.txt
  global-feats-netvlad.h5
  feats-aliked-n16.h5
  feats-aliked-n16_matches-aliked-lightglue_pairs-db-netvlad20.h5

triangulated_model/
  cameras.bin
  images.bin
  points3D.bin
  frames.bin
  rigs.bin
```

`.h5` 比较大，用了 Git LFS。

## 跑一张图

```bash
python -m localization.aliked_lightglue.localize_query \
  --query-image query.jpg \
  --sfm-model-dir localization/baseline_0416_1/triangulated_model \
  --db-global-feats localization/baseline_0416_1/reference_db/global-feats-netvlad.h5 \
  --db-local-feats localization/baseline_0416_1/reference_db/feats-aliked-n16.h5 \
  --ref-points-2d navigation/baseline_0416_1/topometric/reference_points_2d.csv \
  --output-json output.json
```

输出里的 `x/y` 是平面图坐标，`heading_deg` 是朝向。`pnp_debug.floorplan_xy` 只是调试用，位置还是先看加权平均出来的 `x/y`。

## 如果要重新建库

一般不用重来。如果后面确实要重新跑，可以看这几个：

```text
aliked_lightglue/build_reference_db.py
aliked_lightglue/triangulate_reference_model.py
aliked_lightglue/localize_query.py
```

依赖大概是 COLMAP / pycolmap / hloc / h5py / torch / pillow。

## 后面可能要改的地方

现在的问题主要还是回字形走廊太像了，而且参考库基本是单方向拍的。反向拍或者左右比较对称的位置会容易飘到另一边。

如果继续做，我觉得可以先看候选图到底落在哪里，然后拿 301、307 那几个点做固定测试。后面也可以在容易混的地方补一些反向参考图，这些图先只参与检索和位置加权，不一定非要进 SfM。
