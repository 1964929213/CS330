# 数据说明

本文档说明交付包内的数据组成、路径和维护约束。

## 1. 数据概览

| 项目 | 内容 |
| --- | --- |
| 参考集 | `aliked_reference_set_20260506_919` |
| 参考图数量 | 919 |
| 基线模型 | `aliked_baseline_20260416` |
| 坐标单位 | floorplan 像素 |
| 路径配置 | `config/package_manifest.json` |

## 2. 参考集

```text
data/reference_sets/aliked_reference_set_20260506_919/
  reference_images/
  reference_db/
    global-feats-netvlad.h5
    feats-aliked-n16.h5
  topometric/
    reference_points_2d.csv
    transform.json
    correspondences.csv
```

| 文件 / 目录 | 用途 |
| --- | --- |
| `reference_images/` | 最终参考图片，供复核、问题定位和未来更新使用 |
| `global-feats-netvlad.h5` | 全局检索特征库 |
| `feats-aliked-n16.h5` | 局部特征库 |
| `reference_points_2d.csv` | 参考图到 floorplan 像素坐标的映射 |
| `transform.json` | topometric 变换和 floorplan 引用 |
| `correspondences.csv` | topometric 对应点数据 |

文件名关系：

- `reference_images/` 文件名
- H5 keys
- `reference_points_2d.csv` 中的 `image_name`

三者必须保持一致。参考图片文件名不要修改。

## 3. 基线模型

```text
data/baselines/aliked_baseline_20260416/triangulated_model/
```

用途：

- 供 PnP 估计朝向和诊断信息。
- 不决定最终位置。
- 不限制参考图检索范围。

检索范围以 `global-feats-netvlad.h5` 的 keys 为准，因此未注册到三维模型中的 topometric-only 参考图仍可参与定位位置计算。

## 4. 楼层平面图

```text
data/floorplans/corridor_floorplan_0410.jpg
```

`/api/floorplan` 返回该图片。定位结果中的 `x` / `y` 对应该图片原始像素坐标。

坐标定义：

| 项目 | 定义 |
| --- | --- |
| 原点 | 图片左上角 |
| x 轴 | 向右 |
| y 轴 | 向下 |
| 单位 | 像素 |

## 5. 完整性要求

| 检查项 | 数量 / 要求 |
| --- | ---: |
| reference images | 919 |
| `reference_points_2d.csv` rows | 919 |
| NetVLAD H5 keys | 919 |
| ALIKED H5 keys | 919 |
| 文件名集合 | 完全一致 |

## 6. 不应放入交付包的内容

- 源视频。
- 标注或抽帧中间文件。
- 过程实验目录。
- 在线定位生成的 query、match H5。
- 服务日志。
- 本地 TLS 证书或私钥。
- Python 缓存。
- 备份 H5 或特征追加过程日志。

## 7. 更新参考集时的原则

1. 保持图片文件名与 H5 keys、CSV `image_name` 一致。
2. 同步更新两个 H5 和 topometric CSV。
3. floorplan 保持包内相对路径。
4. 更新后重新检查数量与文件名集合一致性。
