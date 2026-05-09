# API 说明

本文档说明定位服务对导航和 Web 前端暴露的接口。接口只描述集成所需字段，不包含服务部署细节。

## 1. 接口列表

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/status` | 返回服务状态和核心数据是否存在 |
| `GET` | `/api/floorplan` | 返回楼层平面图图片 |
| `POST` | `/api/localize` | 上传一张图片并返回定位结果 |

## 2. `GET /api/status`

用于判断服务和数据是否可用。该接口不执行定位。

主要字段：

| 字段 | 说明 |
| --- | --- |
| `ready` | 必要数据文件都存在时为 `true` |
| `paths` | 核心数据路径及存在状态 |
| `last_result` | 最近一次定位结果 |
| `last_error` | 最近一次错误信息 |
| `last_duration_ms` | 最近一次定位耗时 |
| `busy_dropped_count` | 因 busy 被拒绝的请求数量 |
| `is_localizing` | 当前是否正在定位 |
| `localizer_loaded` | 定位模型是否已经加载 |

示例：

```json
{
  "ready": true,
  "paths": {
    "db_global_feats": {"exists": true},
    "db_local_feats": {"exists": true},
    "sfm_model_dir": {"exists": true},
    "ref_points_2d": {"exists": true},
    "transform_json": {"exists": true},
    "floorplan": {"exists": true}
  },
  "is_localizing": false,
  "localizer_loaded": false
}
```

## 3. `GET /api/floorplan`

返回当前定位坐标对应的楼层平面图。`/api/localize` 输出的 `x` / `y` 是这张图的原始像素坐标。

可能响应：

| 状态码 | 说明 |
| --- | --- |
| `200` | 返回图片 |
| `404` | floorplan 缺失或未配置 |

## 4. `POST /api/localize`

输入一张图片，返回定位结果。

请求体：

```json
{
  "image": "data:image/jpeg;base64,/9j/..."
}
```

并发约束：服务一次只处理一个定位请求。如果前一个请求尚未结束，新请求返回 `429 busy`。调用方应丢弃该帧，不要排队。

典型响应：

```json
{
  "ok": true,
  "result": {
    "success": true,
    "x": 123.0,
    "y": 456.0,
    "heading_deg": 90.0,
    "confidence": 0.72,
    "selected_k": 20,
    "num_matches_best": 143,
    "best_candidate": "frame_000123.jpg",
    "inliers_pnp": 45,
    "failure_reason": null,
    "position_debug": {
      "weighted_xy": [123.0, 456.0],
      "pnp_floor_xy": [120.0, 450.0],
      "candidate_dispersion_px": 35.0
    }
  }
}
```

## 5. 结果字段

| 字段 | 说明 | 接入建议 |
| --- | --- | --- |
| `success` | 是否得到有效 topometric 位置 | 为 `true` 时可更新位置 |
| `x`, `y` | floorplan 原始像素坐标 | 导航和地图 UI 使用 |
| `heading_deg` | 可选朝向 | 为空时不要强行显示方向 |
| `confidence` | 诊断参考值 | 不单独作为正确性依据 |
| `selected_k` | 使用的候选数量 | 诊断用 |
| `best_candidate` | 匹配数量最高的参考图 | 诊断用 |
| `inliers_pnp` | PnP 内点数 | 判断朝向可靠性 |
| `failure_reason` | 失败原因 | 日志或提示用 |
| `position_debug.weighted_xy` | 加权 topometric 位置 | 最终位置来源 |
| `position_debug.pnp_floor_xy` | PnP 投影位置 | 仅诊断，不替代 `x` / `y` |
| `position_debug.candidate_dispersion_px` | 候选离散程度 | 用于跳点判断 |

## 6. 常见响应

| 情况 | 状态码 | 说明 |
| --- | --- | --- |
| 定位完成 | `200` | `ok=true`，读取 `result` |
| 服务 busy | `429` | 丢弃当前帧，稍后再请求 |
| 请求格式错误 | `500` | 检查 `image` 字段和 data URL 格式 |
| 数据或模型异常 | `500` | 查看 `error` 和 `/api/status` |

## 7. 解释规则

- 最终位置只使用 `x` / `y`。
- `heading_deg` 可以为空。
- PnP 字段只用于朝向和诊断。
- `confidence` 需要结合候选分布、历史轨迹和导航约束使用。
