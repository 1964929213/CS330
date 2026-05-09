# 导航与 Web 接入说明

本文件说明定位服务和导航 / Web 侧的分工，以及前端显示定位结果时需要注意的坐标与状态处理。

## 1. 定位服务输出

定位服务把单张相机图像转换为 floorplan 上的位置。

| 输出 | 说明 |
| --- | --- |
| `x`, `y` | floorplan 原始像素坐标，导航和地图显示使用 |
| `heading_deg` | 可选朝向，可能为空 |
| `confidence` | 诊断参考值 |
| `position_debug` | 候选位置、离散程度、PnP 对照等诊断信息 |

## 2. 分工边界

定位包负责：

- 接收单帧图片。
- 检索参考图。
- 执行特征匹配。
- 输出 floorplan 坐标。
- 在可用时输出朝向。
- 返回必要诊断信息。

导航 / Web 侧负责：

- 摄像头权限和视频流选择。
- 控制图片上传频率。
- busy 时丢弃当前帧，不排队。
- 将 floorplan 像素坐标映射到页面显示坐标。
- 路径规划、轨迹平滑、跳点处理和 UI 展示。

## 3. 坐标换算

定位结果坐标基于 `/api/floorplan` 返回图片的原始像素尺寸。

| 项目 | 定义 |
| --- | --- |
| 原点 | floorplan 左上角 |
| x 轴 | 向右 |
| y 轴 | 向下 |
| 单位 | 像素 |

如果页面缩放显示 floorplan：

```text
screen_x = result.x * rendered_floorplan_width / natural_floorplan_width
screen_y = result.y * rendered_floorplan_height / natural_floorplan_height
```

如果页面还做了平移、裁剪或 CSS transform，需要在上述换算后继续应用对应变换。

## 4. 请求节奏

建议客户端保持简单策略：

1. 确认 `/api/status` 的 `ready=true`。
2. 加载 `/api/floorplan`。
3. 从视频流中选取清晰帧。
4. 一个定位请求完成前，不发送下一帧。
5. 收到 `429 busy` 时丢弃该帧。
6. `success=true` 时更新位置。
7. `heading_deg` 非空时更新朝向。

## 5. 结果使用建议

| 信号 | 建议 |
| --- | --- |
| `success=false` | 不更新当前位置，可保持上一有效位置 |
| `heading_deg=null` | 只更新位置，不更新朝向 |
| `confidence` 低 | 降低显示可信度或等待下一帧 |
| `candidate_dispersion_px` 高 | 结合历史轨迹或路径约束抑制跳点 |
| `429 busy` | 丢弃当前帧，不重试同一帧 |

## 6. 重要约束

- 最终位置只使用 `x` / `y`。
- PnP 只用于朝向和诊断，不替代最终位置。
- `confidence` 不是绝对正确性保证。
- 重复纹理或对称走廊中，应由导航层结合时序和路径信息做稳定化。
