---
name: video-animation-spec
description: >
  将自然语言或产品需求解析为视频/动画生成用的结构化 JSON 指令（角色、动作、镜头、场景交互），
  判断是否需要拆分为多子场景，为每个子场景编写正向/负向提示词，并汇总到 Markdown 文档。
  只要用户提到视频动画、分镜、镜头语言、角色动作、文生视频、图生视频、动画提示词、
  scene prompt、产品动画需求拆解，即使没说「JSON」或「skill」也应使用本 skill。
---

# 视频动画规格生成（Video Animation Spec）

把用户的**文字描述**和/或**参考图像说明**转成可交给视频模型（文生视频、图生视频、动画生成）的**结构化指令**。交付物是一份 Markdown，内含一个或多个 JSON 代码块。

## 何时触发本 skill

- 用户要「根据描述生成视频/动画」
- 用户给产品 PRD、旁白脚本、分镜草稿，需要变成可执行的生成提示
- 用户问要不要拆场景、怎么写镜头/动作提示词

## 核心流程（按顺序执行）

### 1. 理解输入

收集并显式列出：

| 项 | 说明 |
|----|------|
| 目标 | 广告片 / 教程 / 社交短视频 / 游戏过场等 |
| 时长 | 总时长或「未指定」 |
| 画幅 | 16:9 / 9:16 / 1:1 等 |
| 风格 | 写实 / 3D / 二次元 / 像素等 |
| 参考图 | 有则描述每张图对应角色/场景/构图；无则标注「纯文本」 |
| 禁忌 | 品牌、血腥、变形等必须避免的元素 |

信息不足时，在 Markdown 开头的「假设与待确认」小节列出合理默认，**不要**静默编造关键剧情。

### 2. 判断：单场景 vs 多子场景

**默认单场景**仅当同时满足：动作单一、时间线连续、机位变化可用一条 prompt 说清、总时长通常 ≤5s。

**必须拆为多子场景**若出现任一情况：

- 明显分镜/转场（切镜、淡入淡出、「然后」「接着」）
- 多地点或多时间段
- 多个独立动作节拍（如：助跑 → 踢球 → 球入网 → 庆祝）
- 总时长 >8s 或用户明确要求分段生成
- 不同角色在不同镜头中承担主叙事

拆分后：每个子场景一个独立 JSON；在 Markdown 中说明**场景顺序**与**前后衔接**（上一镜末帧 ↔ 下一镜首帧建议）。

### 3. 解析为结构化 JSON

每个子场景使用统一 schema（字段可省略，但须在 Markdown 里说明省略原因）。完整字段说明见 `references/schema.md`。

**每个场景 JSON 顶层字段：**

- `scene_id`：字符串，如 `scene_01`
- `title`：简短中文标题
- `duration_seconds`：建议时长（数字）
- `aspect_ratio`：如 `"16:9"`
- `style`：整体视觉风格一句话
- `keyframe_image`：首帧/参考**静图**生成用（对象，见下）
- `animation`：**动画/视频**生成用（对象，见下）
- `positive_prompt`：英文为主，逗号分隔，给视频模型
- `negative_prompt`：英文为主，排除常见问题

**`keyframe_image` 对象**（静图 / 垫图）：

- `subject`：角色外观（年龄、服装、发型、道具）
- `pose`：静态姿势
- `environment`：场景、天气、时间
- `composition`：景别、机位高度、主体在画面中的位置
- `lighting`：主光方向、氛围
- `prompt_en`：合并后的英文静图 prompt

**`animation` 对象**（动态）：

- `character`：角色身份与外观要点（与 keyframe 一致）
- `character_action`：时序动作（开始→过程→结束），用动词短语
- `camera`：镜头类型（wide/medium/close）、运动（static/pan/dolly/track/orbit）、速度
- `scene_interaction`：与环境/道具/其他角色的交互
- `motion_intensity`：`low` | `medium` | `high`
- `temporal_notes`：节奏、慢动作、循环等

### 4. 正向 / 负向提示词

- **positive_prompt**：在 `animation` 语义基础上，补全画质与风格词（如 `cinematic lighting, smooth motion, consistent character`）。避免与 `negative_prompt` 矛盾。
- **negative_prompt**：通用底片 + 场景特有问题，例如：`blurry, jitter, flicker, deformed limbs, extra fingers, morphing face, text watermark, logo, low resolution, inconsistent character`。

视频模型常对英文更稳；用户要求中文时，`prompt_en` 仍写英文，可在 Markdown 加「中文摘要」列。

### 5. 输出 Markdown 文档

写入用户指定路径；未指定则：`animation_spec_<简短主题>.md`（放在当前工作目录或用户项目 `docs/`）。

**必须遵循** `assets/output-template.md` 的章节结构。每个场景的 JSON 放在 `### 场景 N` 下的 fenced code block，语言标记为 `json`。

### 6. 质量自检（输出前默念）

- [ ] 是否明确写了「单场景 / 共 N 场景」及拆分理由
- [ ] 每个 JSON 是否同时包含 `keyframe_image` 与 `animation`
- [ ] 动作是否可拍摄（避免「内心感动」等无法成像的描述）
- [ ] 镜头运动是否与动作强度匹配
- [ ] 多场景时是否有衔接说明

## 示例（必读）

完整「踢足球」单场景 → 可扩展为三场景拆分的示例见：

`references/example-football-kick.md`

生成用户文档时，结构应对齐该示例；不要照抄剧情，除非用户明确要求足球场景。

## 参考文件

| 文件 | 何时读 |
|------|--------|
| `references/schema.md` | 需要字段定义或校验 JSON 完整性时 |
| `references/example-football-kick.md` | 首次使用本 skill 或用户要示例时 |
| `assets/output-template.md` | 撰写最终 Markdown 前 |

## 与用户沟通

完成后用简短中文说明：

1. 拆了几段场景、为什么
2. 输出文件路径
3. 待用户确认的假设（若有）

不要代替用户调用视频 API，除非用户明确要求执行生成。
