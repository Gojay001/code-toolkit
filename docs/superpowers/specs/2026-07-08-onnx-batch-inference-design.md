# ONNX 批量推理脚本设计

**日期:** 2026-07-08  
**状态:** Approved

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
src/onnx_preprocess.py  ←── src/onnx_configs.py (per-model config)
        │                           │
        │                   src/alignment/*
        │                   src/detector/  (bvt + dlib，按平台选用)
        ▼
   NCHW float32 tensors
        │
        ▼
src/run_all_onnx.py  ──► onnxruntime InferenceSession
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
├── src/                    # 全部脚本代码
│   ├── alignment/          # 从 ~/Documents/gan_effect vendoring
│   │   ├── face_alignment.py
│   │   ├── beard_alignment.py
│   │   ├── eyebrow_alignment.py
│   │   ├── eyelid_alignment.py
│   │   └── org_alignment.py                 # dlib 通用人脸对齐回退
│   ├── detector/
│   │   ├── bvt_face_detector.py
│   │   ├── landmarks_detector.py              # dlib，非 Linux 回退
│   │   └── shape_predictor_68_face_landmarks.dat
│   ├── onnx_configs.py     # MODEL_CONFIGS
│   ├── onnx_preprocess.py  # 统一前处理
│   ├── run_all_onnx.py     # CLI 入口
│   └── __init__.py         # 包标识（可选，便于 import）
├── models/*.onnx
├── imgs/0.png, 22.png
└── results/
    ├── {model_name}.png | .json
    ├── _preprocessed/
    └── MODELS.md
```

**Vendoring 范围：** alignment 模块 5 个（含 `org_alignment.py` 作 dlib 回退）+ detector 2 个 + dlib dat 文件。

**Import 路径：** 全部脚本放在 `src/` 下；vendored 模块用包内相对 import，或从项目根目录以 `python src/run_all_onnx.py` 运行并在入口添加 repo root 到 `sys.path`。

## 依赖与平台策略

| 环境 | 检测/对齐 | 安装方式 |
|------|-----------|----------|
| Linux | BVT + 模型专用 alignment | 若 `import BVT` 失败，自动 `pip install 3rdparty/BVT-1.48.0-cp310-cp310-linux_x86_64.whl` |
| 非 Linux（macOS 等） | dlib + alignment（`detector='dlib'`） | `pip install dlib`；dat 文件随 repo 提供 |

- Wheel 路径：`3rdparty/BVT-1.48.0-cp310-cp310-linux_x86_64.whl`
- Python 版本：cp310（与 BVT wheel 匹配）
- BVT 仅 Linux x86_64；非 Linux 不尝试安装 BVT

**dlib 回退说明：** `beard` / `face` / `eyelid` 的 alignment 模块源码仅支持 BVT。非 Linux 下对这些模型使用 `org_alignment.image_align_run(..., detector='dlib')` 作通用人脸对齐回退；`eyebrow` 可直接用 `eyebrow_alignment(..., detector='dlib')`。BVT 与 dlib 对齐结果可能有差异，文档中需记录实际使用的 detector。

## 模型配置 Schema

`src/onnx_configs.py` 中每个模型一条 dict：

| 字段 | 类型 | 说明 |
|------|------|------|
| `align` | str | `beard` / `eyebrow` / `eyelid` / `face` / `none` |
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

**eyebrow：** Linux 用 BVT + `detector='bvt'`；非 Linux 用 dlib + `detector='dlib'`。

## 预处理流水线

统一入口：`preprocess_model(config, imgs_dir, preprocessed_dir) -> dict[str, np.ndarray]`

### 决策流程（每张输入图独立判断）

```
加载 PIL RGB 图像
        │
        ▼
图像 (W,H) 是否 == config.input_size 的 (W,H)？
        │
   是 ──┴── 否
   │         │
   │         ▼
   │    sys.platform.startswith("linux")？
   │         │
   │    是 ──┴── 否
   │    │         │
   │    │         ▼
   │    │    dlib LandmarksDetector 检测
   │    │    按 align 类型对齐（见下表）
   │    │
   │    ▼
   │    ensure_bvt_installed()   # pip install 3rdparty wheel
   │    BVTFaceDetector 检测
   │    按 align 类型对齐（模型专用 alignment）
   │
   ▼
跳过 alignment / 裁剪
        │
        ▼
（若尺寸仍与 input_size 不完全一致则 PIL resize）
        │
        ▼
Normalize → NCHW float32
        │
        ▼
保存中间图到 results/_preprocessed/
```

**尺寸匹配规则：** `config.input_size = (H, W)`，PIL `Image.size = (W, H)`；两者宽高分别相等视为匹配，跳过 alignment。`align=none`（gender_race）同样适用：128×128 直接 normalize，否则 resize 后 normalize。

**face_swap：** source / target 各自独立走上述流程。

### 对齐路由表

| align | Linux (BVT) | 非 Linux (dlib) |
|-------|---------------|-----------------|
| beard | `beard_alignment.image_align_run(..., detector='bvt')` | `org_alignment.image_align_run(..., detector='dlib')` |
| eyebrow | `eyebrow_alignment(..., detector='bvt')` | `eyebrow_alignment(..., detector='dlib')` |
| eyelid | `eyelid_alignment` 左右眼 crop + 拼接 | `org_alignment` 全脸对齐到 `align_size`，再 resize 到 `input_size` |
| face | `face_alignment(..., detector='bvt')` | `org_alignment(..., detector='dlib')` |
| none | 不检测、不对齐 | 不检测、不对齐 |

### 步骤摘要

1. **加载图像** — PIL RGB
2. **尺寸检查** — 匹配则跳过检测与 alignment
3. **平台分支** — Linux 装/用 BVT；否则 dlib
4. **人脸检测** — 取第一张脸；无脸则 raise，上层 catch 并标记 failed
5. **Alignment** — 按上表路由
6. **Resize** — 必要时 PIL resize 到 `input_size`
7. **Normalize** — HWC → NCHW float32，`minus1_1`
8. **保存中间图** — `results/_preprocessed/{model_name}.png`（face_swap 写 `_source` / `_target` 后缀）；跳过 alignment 时保存原图副本

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
3. **预处理** — 是否跳过 alignment、实际 detector（bvt/dlib/skipped）、align 类型、input_size、normalize、输入图
4. **后处理** — 图像保存路径或 JSON 格式说明
5. **本次运行结果** — success/failed、输出文件路径、错误信息（如有）

## CLI 接口

```bash
python src/run_all_onnx.py \
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
| BVT 安装失败（Linux） | 该模型 failed，日志输出 pip 错误 |
| dlib 未安装（非 Linux） | 该模型 failed，提示 `pip install dlib` |
| ONNX 推理异常 | 捕获异常，继续下一个模型 |
| 未知 .onnx 文件 | warn 并跳过 |
| 输入图不存在 | 启动时 fail fast |

## Python 依赖

```
onnxruntime
numpy
Pillow
scipy
dlib                        # 非 Linux 必须
BVT                         # Linux，从 3rdparty wheel 自动或手动安装
```

`ensure_bvt_installed()` 实现要点：

```python
def ensure_bvt_installed(wheel_path: str) -> None:
    try:
        import BVT
        return
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
```

## 测试验证

1. **Linux cp310**：输入非对齐大图 → 自动装 BVT → 模型专用 alignment → 推理成功
2. **Linux cp310**：输入已是 `input_size` 的图 → 跳过 alignment，仅 normalize
3. **macOS**：输入非对齐大图 → dlib 回退 alignment → 推理成功（eyebrow/face/beard；eyelid 走 org_alignment 回退）
4. 运行 `python src/run_all_onnx.py`
5. 检查：
   - `results/` 下 5 张 png + 1 个 json
   - `results/_preprocessed/` 有中间对齐图
   - `results/MODELS.md` 覆盖 6 个模型的 I/O 与预处理说明

## 实现顺序（供 writing-plans 使用）

1. Vendoring alignment + detector 模块到 `src/`（含 org_alignment、dlib dat）
2. 编写 `src/onnx_configs.py`
3. 编写 `src/onnx_preprocess.py`（尺寸检查 + 平台分支 + ensure_bvt_installed）
4. 编写 `src/run_all_onnx.py`（推理 + 后处理 + 文档生成）
5. Linux 与 macOS 分别端到端验证
