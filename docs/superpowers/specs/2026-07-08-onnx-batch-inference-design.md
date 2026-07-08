# ONNX 批量推理脚本设计

**日期:** 2026-07-08  
**状态:** 待用户审阅

## 目标

在 `code-toolkit` 内实现一个批量 ONNX 推理脚本，对 `models/` 下全部 `.onnx` 模型运行推理，并自动生成模型 I/O 文档。

- 默认输入图：`imgs/22.png`（`face_swap` 的 source 用 `imgs/0.png`，target 用 `imgs/22.png`）
- 输出目录：`results/`
- 预处理逻辑统一实现，通过 per-model 配置区分
- alignment / detector 代码从 `gan_effect` vendoring 到本仓库（方案 B）
- 保存预处理中间对齐图
- `gender_race` 输出 JSON

## 非目标

- 不支持训练或模型转换
- 不实现 Web UI 或 REST API
- 不在本阶段处理视频批量推理

## 架构概览

```
imgs/0.png, 22.png
        │
        ▼
onnx_preprocess.py  ←── onnx_configs.py (per-model config)
        │                      │
        │              source_code/alignment/*
        │              source_code/detector/bvt_face_detector.py
        ▼
   NCHW float32 tensors
        │
        ▼
run_all_onnx.py  ──► onnxruntime InferenceSession
        │
        ├── results/{model}.png / .json
        ├── results/_preprocessed/{model}_*.png
        └── results/MODELS.md
```

## 目录结构

```
code-toolkit/
├── 3rdparty/
│   └── BVT-1.48.0-cp310-cp310-linux_x86_64.whl   # BVT SDK
├── source_code/
│   ├── alignment/          # 从 ~/Documents/gan_effect vendoring
│   │   ├── face_alignment.py
│   │   ├── beard_alignment.py
│   │   ├── eyebrow_alignment.py
│   │   └── eyelid_alignment.py
│   └── detector/
│       └── bvt_face_detector.py
├── utils/
│   ├── onnx_configs.py     # MODEL_CONFIGS
│   ├── onnx_preprocess.py  # 统一前处理
│   └── run_all_onnx.py     # CLI 入口
├── models/*.onnx
├── imgs/0.png, 22.png
└── results/
    ├── {model_name}.png | .json
    ├── _preprocessed/
    └── MODELS.md
```

**Vendoring 范围：** 仅拷贝上述 5 个 Python 文件；不拷贝 `landmarks_detector.py`（全部模型使用 BVT 检测）。

**Import 路径：** vendored 模块放在 `source_code/` 下，推理脚本通过 `sys.path` 或相对 import 引用，保持与 `gan_effect` 内模块逻辑一致。

## BVT 依赖

- Wheel 路径：`3rdparty/BVT-1.48.0-cp310-cp310-linux_x86_64.whl`
- 安装方式：文档/脚本说明中要求 `pip install 3rdparty/BVT-1.48.0-cp310-cp310-linux_x86_64.whl`
- Python 版本：cp310（与 wheel 匹配）
- 平台：linux x86_64（当前 wheel 为 Linux 构建；macOS 本地无法直接安装该 wheel，需在 Linux cp310 环境运行）

## 模型配置 Schema

`utils/onnx_configs.py` 中每个模型一条 dict：

| 字段 | 类型 | 说明 |
|------|------|------|
| `align` | str | `beard` / `eyebrow` / `eyelid` / `face` / `none` |
| `detector` | str | 固定 `bvt` |
| `align_size` | int | `image_align_run(..., output_size=...)` |
| `input_size` | (H, W) | 对齐后二次 resize 到模型输入尺寸 |
| `normalize` | str | `minus1_1` → `(x/255)*2-1` |
| `inputs` | list[str] | 相对 `imgs/` 的文件名 |
| `input_names` | list[str] | ONNX 输入 tensor 名 |
| `output_type` | str | `image` 或 `classification` |
| `output_channels` | int | 图像后处理通道数（3 或 4） |
| `output_names` | list[str] | 可选，classification 必填 |

### 各模型预设

| 模型 | align | align_size | input_size | inputs | input_names | output |
|------|-------|------------|------------|--------|-------------|--------|
| beard_eliminate | beard | 256 | (192, 256) | [22.png] | [input] | image, 4ch |
| eyebrow_eliminate | eyebrow | 256 | (128, 256) | [22.png] | [input] | image, 4ch |
| eyelid_double2single | eyelid | 128 | (128, 128) | [22.png] | [input] | image, 3ch |
| eyelid_single2double | eyelid | 128 | (128, 128) | [22.png] | [input] | image, 3ch |
| face_swap | face | 256 | (256, 256) | [0.png, 22.png] | [source, target] | image, 4ch |
| gender_race | none | — | (128, 128) | [22.png] | [input] | classification |

**face_swap 映射：** `inputs[0]` → `source`，`inputs[1]` → `target`。

**eyelid 特殊逻辑：** 对单张图检测 BVT landmarks 后，分别 crop 左右眼（`crop_eye='left_eye'` / `'right_eye'`，`output_size=128`），上下拼接为 128×128（上左眼下右眼），与 `align_images_for_eyelid.py` 一致。

**gender_race：** 不做 face alignment，仅 resize 到 128×128 + normalize。

**eyebrow：** 使用 BVT detector + `eyebrow_alignment.image_align_run(..., detector='bvt')`；推理时只需单张输入图（无 mask/gt 分支），对齐后 resize 到 (128, 256)。

## 预处理流水线

统一入口：`preprocess_model(config, imgs_dir, preprocessed_dir) -> dict[str, np.ndarray]`

1. **加载图像** — PIL RGB
2. **人脸检测** — `BVTFaceDetector.get_landmarks()`，取第一张脸；无脸则 raise，上层 catch 并标记 failed
3. **Alignment** — 按 `config.align` 调用对应 `image_align_run`
4. **Resize** — PIL resize 到 `input_size` (H, W)
5. **Normalize** — HWC → NCHW float32，`minus1_1`
6. **保存中间图** — 写入 `results/_preprocessed/{model_name}.png`（face_swap 写 `{model}_source.png` 和 `{model}_target.png`）

## 推理与后处理

### 通用

- 引擎：`onnxruntime`，默认 CPU，`--gpu` 启用 CUDAExecutionProvider
- 遍历 `models/*.onnx`，按文件名匹配 `MODEL_CONFIGS`；未知模型跳过并 warn

### 图像输出 (`output_type=image`)

- 取第一个输出 tensor
- 4 通道：取前 3 通道 RGB，或按现有 `run_face_swap_onnx.py` 逻辑处理
- 反归一化：`(x + 1) * 127.5`，clip 到 [0, 255]
- 保存：`results/{model_stem}.png`

### 分类输出 (`gender_race`)

- 运行 `output_names = ["output_gender", "output_race"]`
- 保存：`results/gender_race.json`

```json
{
  "model": "gender_race.onnx",
  "inputs": {"input": "imgs/22.png"},
  "outputs": {
    "output_gender": {"shape": [1, 1], "value": 0.82},
    "output_race": {"shape": [1, 4], "values": [...], "argmax": 2}
  },
  "status": "ok"
}
```

## 文档生成

运行结束后写 `results/MODELS.md`，每个模型一节：

1. **模型文件** — 路径、文件大小
2. **ONNX I/O** — 从 `session.get_inputs()` / `get_outputs()` 读取 name、shape、dtype
3. **预处理** — 从 config 生成人类可读说明（alignment 类型、align_size、input_size、normalize、输入图）
4. **后处理** — 图像保存路径或 JSON 格式说明
5. **本次运行结果** — success/failed、输出文件路径、错误信息（如有）

## CLI 接口

```bash
python utils/run_all_onnx.py \
  --models-dir models \
  --imgs-dir imgs \
  --results-dir results \
  [--image 22.png] \
  [--source 0.png] \
  [--cpu]
```

默认值与用户需求一致；`--image` / `--source` 可覆盖单图路径。

## 错误处理

| 场景 | 行为 |
|------|------|
| 未检测到人脸 | 跳过该模型，`MODELS.md` 标注 failed |
| ONNX 推理异常 | 捕获异常，继续下一个模型 |
| 未知 .onnx 文件 | warn 并跳过 |
| 输入图不存在 | 启动时 fail fast |

## Python 依赖

```
onnxruntime
numpy
Pillow
scipy
opencv-python   # 可选，若 alignment 模块未用到可不装
BVT             # 从 3rdparty wheel 安装
```

## 测试验证

1. Linux cp310 环境安装 BVT wheel + 依赖
2. 运行 `python utils/run_all_onnx.py`
3. 检查：
   - `results/` 下 5 张 png + 1 个 json
   - `results/_preprocessed/` 有中间对齐图
   - `results/MODELS.md` 覆盖 6 个模型的 I/O 与预处理说明

## 实现顺序（供 writing-plans 使用）

1. Vendoring alignment + detector 模块到 `source_code/`
2. 编写 `onnx_configs.py`
3. 编写 `onnx_preprocess.py`
4. 编写 `run_all_onnx.py`（推理 + 后处理 + 文档生成）
5. 在 Linux 环境端到端验证
