# 响应示例

本文档保留接入时最常用的响应形态，便于前端和导航侧对字段做适配。

## 1. 状态正常

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
  "last_result": null,
  "last_error": null,
  "last_duration_ms": null,
  "busy_dropped_count": 0,
  "is_localizing": false,
  "localizer_loaded": false
}
```

关注点：

- `ready=true`：核心数据存在。
- `localizer_loaded=false`：尚未执行定位，不是错误。

## 2. 服务 busy

```json
{
  "ok": false,
  "error": "busy",
  "status": {
    "ready": true,
    "is_localizing": true,
    "busy_dropped_count": 1
  }
}
```

关注点：

- 通常对应 HTTP `429`。
- 客户端丢弃当前帧，等待下一次采样。

## 3. 定位成功

```json
{
  "ok": true,
  "result": {
    "success": true,
    "x": 123.0,
    "y": 456.0,
    "heading_deg": 87.5,
    "confidence": 0.73,
    "selected_k": 20,
    "num_matches_best": 110,
    "best_candidate": "frame_000123.jpg",
    "inliers_pnp": 31,
    "failure_reason": null,
    "position_debug": {
      "weighted_xy": [123.0, 456.0],
      "pnp_floor_xy": [119.0, 452.0],
      "candidate_dispersion_px": 42.0
    },
    "match_count_by_candidate": {
      "frame_000123.jpg": 110,
      "frame_000124.jpg": 58
    },
    "candidate_floorplan_xy": {
      "frame_000123.jpg": [120.0, 450.0],
      "frame_000124.jpg": [128.0, 465.0]
    }
  }
}
```

关注点：

- 使用 `x` / `y` 更新地图位置。
- `heading_deg` 可为空，使用前需要判断。
- `pnp_floor_xy` 只作为诊断参考。

## 4. 位置成功但无朝向

```json
{
  "ok": true,
  "result": {
    "success": true,
    "x": 123.0,
    "y": 456.0,
    "heading_deg": null,
    "inliers_pnp": 8,
    "position_debug": {
      "weighted_xy": [123.0, 456.0],
      "pnp_floor_xy": null
    }
  }
}
```

关注点：

- 位置仍然有效。
- 不要因为朝向为空而丢弃位置。

## 5. 定位失败

```json
{
  "ok": true,
  "result": {
    "success": false,
    "x": null,
    "y": null,
    "heading_deg": null,
    "confidence": 0.0,
    "selected_k": null,
    "num_matches_best": 12,
    "best_candidate": "frame_000321.jpg",
    "failure_reason": "insufficient_matches_after_k_expansion"
  }
}
```

关注点：

- 不更新当前位置。
- 可以保持上一有效位置，等待下一帧。

## 6. 请求格式错误

```json
{
  "ok": false,
  "error": "missing image",
  "status": {
    "ready": true,
    "is_localizing": false
  }
}
```

关注点：

- 检查请求体是否包含 `image`。
- `image` 应为 `data:image/...;base64,...` 格式。
