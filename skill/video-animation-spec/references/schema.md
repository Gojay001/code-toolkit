# 场景 JSON Schema 说明

每个子场景一个 JSON 对象。多场景时输出 JSON 数组，或在 Markdown 中分多个代码块（推荐分块，便于复制到不同生成任务）。

## 顶层

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `scene_id` | string | 是 | 唯一 ID，如 `scene_01` |
| `title` | string | 是 | 场景标题（中文） |
| `duration_seconds` | number | 是 | 建议片段时长（秒） |
| `aspect_ratio` | string | 否 | 默认 `16:9` |
| `style` | string | 否 | 视觉风格一句话 |
| `keyframe_image` | object | 是 | 静图/首帧规格 |
| `animation` | object | 是 | 动画/视频动态规格 |
| `positive_prompt` | string | 是 | 视频模型正向提示（英文） |
| `negative_prompt` | string | 是 | 视频模型负向提示（英文） |
| `continuity` | object | 否 | 多场景时：与前后镜衔接说明 |

## `keyframe_image`

| 字段 | 类型 | 说明 |
|------|------|------|
| `subject` | string | 角色外观 |
| `pose` | string | 静态姿势 |
| `environment` | string | 场景环境 |
| `composition` | string | 构图与景别 |
| `lighting` | string | 光照 |
| `prompt_en` | string | 合并英文静图 prompt |

## `animation`

| 字段 | 类型 | 说明 |
|------|------|------|
| `character` | string | 角色（与 keyframe 一致） |
| `character_action` | string | 动作时间线 |
| `camera` | string | 镜头类型与运动 |
| `scene_interaction` | string | 场景与道具交互 |
| `motion_intensity` | string | `low` / `medium` / `high` |
| `temporal_notes` | string | 节奏、特效 |

## `continuity`（可选）

| 字段 | 类型 | 说明 |
|------|------|------|
| `prev_scene_end` | string | 上一镜结束状态 |
| `this_scene_start` | string | 本镜开始状态 |
| `match_elements` | string[] | 须保持一致的元素（服装、球号、天气等） |
