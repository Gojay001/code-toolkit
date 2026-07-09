# 示例：踢足球动画

**需求（自然语言）**  
「写实风格，青少年在晴天足球场踢球：助跑、右脚射门、足球飞入网，镜头从侧面跟拍。」

**拆分结论**：可拆 3 子场景（助跑 / 击球瞬间 / 球入网），也可合并为 1 段 5s 短片。下面给出**单场景完整 JSON** + **三场景拆分简表**。

---

## 单场景 JSON（5s 一镜到底）

```json
{
  "scene_id": "scene_01",
  "title": "侧面跟拍踢球入网",
  "duration_seconds": 5,
  "aspect_ratio": "16:9",
  "style": "photorealistic sports footage, daytime",
  "keyframe_image": {
    "subject": "teenage male athlete, red jersey number 10, black shorts, white cleats, short black hair",
    "pose": "right leg drawn back ready to strike, left foot planted, torso slightly leaned back, eyes on ball",
    "environment": "outdoor soccer field, green grass, white goal net in background, clear blue sky",
    "composition": "medium-full shot, subject left of center, ball on grass right of subject, sideline perspective",
    "lighting": "natural sunlight from camera-left, soft shadows on grass",
    "prompt_en": "photorealistic teenage soccer player in red jersey number 10, mid-kick pose on sunny grass field, ball on ground, goal in background, side view, cinematic sports photography, sharp focus, 16:9"
  },
  "animation": {
    "character": "same teenage player in red #10 jersey",
    "character_action": "plants left foot, swings right leg forward to strike ball, follows through with brief balance recovery, gaze tracks ball",
    "camera": "medium tracking shot from sideline, camera pans slightly right following player and ball trajectory, smooth gimbal motion",
    "scene_interaction": "right cleat contacts ball, ball rolls then lifts toward goal, grass blades disturbed on impact",
    "motion_intensity": "high",
    "temporal_notes": "impact frame around 2s, ball flight 2-4s, no slow motion unless specified"
  },
  "positive_prompt": "photorealistic soccer kick, teenage athlete red jersey, outdoor sunny field, ball flying toward goal net, side tracking shot, smooth natural motion, cinematic sports lighting, consistent character, sharp details, 24fps feel",
  "negative_prompt": "blurry, jitter, flicker, deformed legs, extra limbs, morphing face, wrong jersey number, indoor stadium, night scene, cartoon, anime, text watermark, logo, crowd duplication, ball teleporting"
}
```

---

## 三场景拆分（推荐分段生成时）

| scene_id | title | duration | 衔接要点 |
|----------|-------|----------|----------|
| scene_01 | 助跑逼近 | 2s | 结束：右脚后摆、球静止 |
| scene_02 | 击球瞬间 | 1.5s | 开始：接 scene_01 末帧；结束：球刚离脚 |
| scene_03 | 球入网 | 2.5s | 开始：球低空飞向球门；结束：球触网 |

### scene_02 动画 JSON 片段（击球）

```json
{
  "scene_id": "scene_02",
  "title": "右脚击球瞬间",
  "duration_seconds": 1.5,
  "aspect_ratio": "16:9",
  "style": "photorealistic sports, high shutter feel",
  "keyframe_image": {
    "subject": "same player red #10, right leg extended contacting ball",
    "pose": "impact pose, grass spray at contact point",
    "environment": "same soccer field, goal visible",
    "composition": "medium shot, slight low angle emphasizing power",
    "lighting": "same daytime sun",
    "prompt_en": "soccer player striking ball, impact moment, grass particles, low angle medium shot, photorealistic"
  },
  "animation": {
    "character": "same player",
    "character_action": "cleat meets ball, ankle locked, short follow-through, body weight shifts forward",
    "camera": "static medium-low angle, minimal shake on impact",
    "scene_interaction": "ball compresses slightly at contact, grass debris",
    "motion_intensity": "high",
    "temporal_notes": "peak action at 0.5-1.0s"
  },
  "positive_prompt": "soccer kick impact, cleat on ball, grass spray, photorealistic, frozen-action clarity, consistent red jersey player",
  "negative_prompt": "blurry, double ball, deformed foot, cartoon, wrong kit, camera whip pan",
  "continuity": {
    "prev_scene_end": "player right leg cocked back, ball stationary",
    "this_scene_start": "leg accelerating into ball",
    "match_elements": ["jersey number 10", "field", "daytime", "ball design"]
  }
}
```

---

## 使用说明

- **静图任务**：主要用 `keyframe_image.prompt_en` 生成参考图，再垫图做视频。
- **视频任务**：以 `positive_prompt` + `negative_prompt` 为主，把 `animation` 各字段改写进一段连贯英文描述。
- 用户只给文字、不给图时：仍须写满 `keyframe_image`，作为首帧生成规格。
