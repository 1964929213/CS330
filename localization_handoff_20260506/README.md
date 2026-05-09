# ALIKED 室内定位交付说明

本目录整理了当前可用的室内视觉定位数据与最小服务封装，供导航和统一 Web 前端接入。交付包内的数据路径均为相对路径，不依赖研究过程目录或本机绝对路径。

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 交付目录 | `localization_handoff_20260506/` |
| 参考集 | `aliked_reference_set_20260506_919` |
| 基线模型 | `aliked_baseline_20260416` |
| 参考图数量 | 919 |
| 定位输出 | 楼层平面图像素坐标 `x` / `y`，可选 `heading_deg` |
| 坐标原点 | floorplan 左上角 |
| 坐标方向 | x 向右，y 向下 |

## 2. 包内容

```text
localization_handoff_20260506/
  config/package_manifest.json
  service/                         # 定位服务封装
  static/                          # 最小参考页面，仅用于对接参考
  data/
    reference_sets/aliked_reference_set_20260506_919/
      reference_images/            # 参考图，文件名不要修改
      reference_db/                # NetVLAD / ALIKED H5 特征库
      topometric/                  # 参考图到平面图坐标
    baselines/aliked_baseline_20260416/
      triangulated_model/          # PnP 朝向估计使用
    floorplans/
      corridor_floorplan_0410.jpg
  docs/
```

关键数据：

| 数据 | 路径 | 说明 |
| --- | --- | --- |
| 参考图片 | `data/reference_sets/aliked_reference_set_20260506_919/reference_images/` | 919 张，文件名是跨文件关联键 |
| 全局特征库 | `reference_db/global-feats-netvlad.h5` | retrieval 使用 |
| 局部特征库 | `reference_db/feats-aliked-n16.h5` | ALIKED + LightGlue 匹配使用 |
| 平面图坐标 | `topometric/reference_points_2d.csv` | 每张参考图对应 floorplan 像素坐标 |
| 楼层平面图 | `data/floorplans/corridor_floorplan_0410.jpg` | 前端显示与坐标换算基准 |

## 3. 接入接口

定位服务保留三个接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/status` | 查看数据与服务状态 |
| `GET` | `/api/floorplan` | 获取楼层平面图 |
| `POST` | `/api/localize` | 上传单张图像并返回定位结果 |

`POST /api/localize` 输入：

```json
{
  "image": "data:image/jpeg;base64,/9j/..."
}
```

核心输出字段：

| 字段 | 说明 |
| --- | --- |
| `success` | 是否得到有效位置 |
| `x`, `y` | 导航与地图显示使用的最终位置 |
| `heading_deg` | 可选朝向，可能为空 |
| `confidence` | 诊断参考值，不应单独作为正确性判断 |
| `position_debug` | 候选、加权位置、PnP 对照等诊断信息 |

接口细节见 `docs/api_contract.md` 和 `docs/response_examples.md`。

## 4. 定位结果解释

- `x` / `y` 是 floorplan 原始像素坐标。
- 最终位置来自 topometric 加权结果。
- PnP 只用于 `heading_deg` 和诊断字段，不替代最终 `x` / `y`。
- `heading_deg` 为空时，仍可使用 `x` / `y` 更新位置。
- 走廊重复纹理或对称区域可能出现跳点，建议导航层结合历史轨迹、路径约束和候选离散度处理。

前端坐标换算：

```text
screen_x = result.x * rendered_floorplan_width / natural_floorplan_width
screen_y = result.y * rendered_floorplan_height / natural_floorplan_height
```

## 5. 集成分工

定位包负责：

- 接收单帧图像。
- 检索参考图并完成特征匹配。
- 返回 floorplan 坐标、可选朝向和诊断信息。
- 在已有定位请求运行时返回 busy，避免请求堆积。

导航 / Web 侧负责：

- 摄像头权限、视频流选择和前端安全上下文。
- 控制请求频率，busy 时丢弃当前帧。
- 将 floorplan 像素坐标映射到页面显示坐标。
- 路径规划、位置平滑、跳点处理和 UI 展示。

## 6. 数据维护注意事项

- 不要重命名 `reference_images/` 内的图片。
- 更新参考集时，参考图、两个 H5、topometric CSV 必须同步更新，并保持文件名集合一致。
- floorplan 保持在包内，路径通过 `config/package_manifest.json` 管理。
- 不要把源视频、标注中间帧、运行日志、在线 query、live H5、证书私钥或 Python 缓存放入交付包。

## 7. 相关文档

- `docs/data_manifest.md`：数据资产说明。
- `docs/api_contract.md`：接口字段与响应约定。
- `docs/navigation_integration.md`：导航与 Web 接入边界。
- `docs/response_examples.md`：典型响应示例。
