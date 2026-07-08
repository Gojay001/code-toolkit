# ONNX 批量推理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `src/run_all_onnx.py`，对 `models/` 下 6 个 ONNX 模型批量推理，按平台/尺寸智能预处理，输出结果图与 `results/MODELS.md`。

**Architecture:** 从 `gan_effect` vendoring alignment/detector 到 `src/`；`onnx_configs.py` 定义 per-model 配置；`onnx_preprocess.py` 统一前处理（尺寸匹配跳过 alignment、Linux 自动装 BVT、dlib 回退）；`run_all_onnx.py` 负责推理、后处理、文档生成。

**Tech Stack:** Python 3.10, onnxruntime, numpy, Pillow, scipy, dlib, BVT (Linux wheel)

**Spec:** `docs/superpowers/specs/2026-07-08-onnx-batch-inference-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/__init__.py` | 包标识 |
| `src/alignment/*.py` | vendored 对齐逻辑（5 文件） |
| `src/detector/bvt_face_detector.py` | BVT 人脸检测 |
| `src/detector/landmarks_detector.py` | dlib 68 点检测 |
| `src/detector/shape_predictor_68_face_landmarks.dat` | dlib 模型权重 |
| `src/onnx_configs.py` | 6 模型配置 dict |
| `src/onnx_preprocess.py` | 尺寸检查、平台分支、alignment 路由、normalize |
| `src/run_all_onnx.py` | CLI、推理循环、后处理、MODELS.md |
| `tests/test_onnx_preprocess.py` | 纯函数单元测试（无 BVT/dlib） |
| `requirements-onnx.txt` | 依赖清单 |

---

### Task 1: 项目脚手架与依赖

**Files:**
- Create: `src/__init__.py`
- Create: `requirements-onnx.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_onnx_preprocess.py`

- [ ] **Step 1: 创建空包与 requirements**

`src/__init__.py`:
```python
# ONNX batch inference package
```

`requirements-onnx.txt`:
```
onnxruntime>=1.16.0
numpy>=1.24.0
Pillow>=10.0.0
scipy>=1.11.0
dlib>=19.24.0
pytest>=7.4.0
```

- [ ] **Step 2: 创建测试文件骨架**

`tests/test_onnx_preprocess.py`:
```python
import numpy as np
from PIL import Image

from src.onnx_preprocess import image_matches_input_size, pil_to_nchw_minus1_1


def test_image_matches_input_size_exact():
    img = Image.new("RGB", (256, 192))  # W=256, H=192
    assert image_matches_input_size(img, (192, 256)) is True


def test_image_matches_input_size_mismatch():
    img = Image.new("RGB", (512, 512))
    assert image_matches_input_size(img, (192, 256)) is False


def test_pil_to_nchw_minus1_1_shape_and_range():
    img = Image.new("RGB", (128, 128), color=(255, 0, 0))
    arr = pil_to_nchw_minus1_1(img)
    assert arr.shape == (1, 3, 128, 128)
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0 and arr.min() >= -1.0
```

- [ ] **Step 3: 运行测试（预期 FAIL — 模块未实现）**

```bash
cd /Users/bigo10295/Downloads/code-toolkit
python -m pytest tests/test_onnx_preprocess.py -v
```
Expected: `ModuleNotFoundError: src.onnx_preprocess`

- [ ] **Step 4: Commit**

```bash
git add src/__init__.py requirements-onnx.txt tests/
git commit -m "chore: scaffold src package and onnx test skeleton"
```

---

### Task 2: Vendoring alignment 模块

**Files:**
- Create: `src/alignment/__init__.py`
- Create: `src/alignment/face_alignment.py` (copy from gan_effect)
- Create: `src/alignment/beard_alignment.py`
- Create: `src/alignment/eyebrow_alignment.py`
- Create: `src/alignment/eyelid_alignment.py`
- Create: `src/alignment/org_alignment.py`

- [ ] **Step 1: 拷贝 alignment 文件**

```bash
GAN=/Users/bigo10295/Documents/gan_effect
DST=/Users/bigo10295/Downloads/code-toolkit/src/alignment
mkdir -p "$DST"
cp "$GAN/src/alignment/face_alignment.py" "$DST/"
cp "$GAN/src/alignment/beard_alignment.py" "$DST/"
cp "$GAN/src/alignment/eyebrow_alignment.py" "$DST/"
cp "$GAN/src/alignment/eyelid_alignment.py" "$DST/"
cp "$GAN/src/alignment/org_alignment.py" "$DST/"
touch "$DST/__init__.py"
```

- [ ] **Step 2: 验证 import 无 `src.` 前缀依赖**

```bash
grep -r "from src\|import src" src/alignment/ || echo "OK: no src imports"
```

- [ ] **Step 3: Commit**

```bash
git add src/alignment/
git commit -m "feat: vendor alignment modules from gan_effect"
```

---

### Task 3: Vendoring detector 模块与 dlib dat

**Files:**
- Create: `src/detector/__init__.py`
- Create: `src/detector/bvt_face_detector.py`
- Create: `src/detector/landmarks_detector.py`
- Create: `src/detector/shape_predictor_68_face_landmarks.dat`

- [ ] **Step 1: 拷贝 detector Python 文件**

```bash
GAN=/Users/bigo10295/Documents/gan_effect
DST=/Users/bigo10295/Downloads/code-toolkit/src/detector
mkdir -p "$DST"
cp "$GAN/src/detector/bvt_face_detector.py" "$DST/"
cp "$GAN/src/detector/landmarks_detector.py" "$DST/"
touch "$DST/__init__.py"
```

- [ ] **Step 2: 下载 dlib shape predictor（若本地不存在）**

```bash
DAT=src/detector/shape_predictor_68_face_landmarks.dat
if [ ! -f "$DAT" ]; then
  curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -o /tmp/shape.dat.bz2
  bunzip2 -c /tmp/shape.dat.bz2 > "$DAT"
fi
ls -lh "$DAT"
```
Expected: ~99MB dat 文件

- [ ] **Step 3: Commit（dat 文件较大，若 git 不便可加入 .gitignore 并在 README 说明下载步骤；默认尝试 commit）**

```bash
git add src/detector/bvt_face_detector.py src/detector/landmarks_detector.py src/detector/__init__.py
git commit -m "feat: vendor BVT and dlib detector modules"
# dat 单独 commit 或文档说明手动下载
```

---

### Task 4: 模型配置

**Files:**
- Create: `src/onnx_configs.py`
- Modify: `tests/test_onnx_preprocess.py`

- [ ] **Step 1: 编写配置**

`src/onnx_configs.py`:
```python
MODEL_CONFIGS = {
    "beard_eliminate": {
        "align": "beard",
        "align_size": 256,
        "input_size": (192, 256),
        "normalize": "minus1_1",
        "inputs": ["22.png"],
        "input_names": ["input"],
        "output_type": "image",
        "output_channels": 4,
    },
    "eyebrow_eliminate": {
        "align": "eyebrow",
        "align_size": 256,
        "input_size": (128, 256),
        "normalize": "minus1_1",
        "inputs": ["22.png"],
        "input_names": ["input"],
        "output_type": "image",
        "output_channels": 4,
    },
    "eyelid_double2single": {
        "align": "eyelid",
        "align_size": 128,
        "input_size": (128, 128),
        "normalize": "minus1_1",
        "inputs": ["22.png"],
        "input_names": ["input"],
        "output_type": "image",
        "output_channels": 3,
    },
    "eyelid_single2double": {
        "align": "eyelid",
        "align_size": 128,
        "input_size": (128, 128),
        "normalize": "minus1_1",
        "inputs": ["22.png"],
        "input_names": ["input"],
        "output_type": "image",
        "output_channels": 3,
    },
    "face_swap": {
        "align": "face",
        "align_size": 256,
        "input_size": (256, 256),
        "normalize": "minus1_1",
        "inputs": ["0.png", "22.png"],
        "input_names": ["source", "target"],
        "output_type": "image",
        "output_channels": 4,
    },
    "gender_race": {
        "align": "none",
        "align_size": None,
        "input_size": (128, 128),
        "normalize": "minus1_1",
        "inputs": ["22.png"],
        "input_names": ["input"],
        "output_type": "classification",
        "output_names": ["output_gender", "output_race"],
    },
}


def get_config_for_model(model_path: str) -> dict | None:
    stem = model_path.rsplit("/", 1)[-1].replace(".onnx", "")
    return MODEL_CONFIGS.get(stem)
```

- [ ] **Step 2: 添加配置测试**

在 `tests/test_onnx_preprocess.py` 追加:
```python
from src.onnx_configs import get_config_for_model, MODEL_CONFIGS


def test_get_config_for_model_face_swap():
    cfg = get_config_for_model("models/face_swap.onnx")
    assert cfg["input_names"] == ["source", "target"]
    assert cfg["inputs"] == ["0.png", "22.png"]


def test_all_onnx_models_have_config():
    expected = {
        "beard_eliminate", "eyebrow_eliminate",
        "eyelid_double2single", "eyelid_single2double",
        "face_swap", "gender_race",
    }
    assert set(MODEL_CONFIGS.keys()) == expected
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_onnx_preprocess.py::test_get_config_for_model_face_swap -v
python -m pytest tests/test_onnx_preprocess.py::test_all_onnx_models_have_config -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/onnx_configs.py tests/test_onnx_preprocess.py
git commit -m "feat: add per-model ONNX configs"
```

---

### Task 5: 预处理工具函数

**Files:**
- Create: `src/onnx_preprocess.py`（第一部分：纯函数）
- Modify: `tests/test_onnx_preprocess.py`

- [ ] **Step 1: 实现纯函数**

`src/onnx_preprocess.py` 开头部分:
```python
import os
import sys
import subprocess
import platform
from typing import Tuple

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BVT_WHEEL = os.path.join(REPO_ROOT, "3rdparty", "BVT-1.48.0-cp310-cp310-linux_x86_64.whl")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def image_matches_input_size(img: Image.Image, input_size: Tuple[int, int]) -> bool:
    h, w = input_size
    return img.size == (w, h)


def resize_to_input_size(img: Image.Image, input_size: Tuple[int, int]) -> Image.Image:
    h, w = input_size
    if img.size == (w, h):
        return img
    return img.resize((w, h), Image.LANCZOS)


def pil_to_nchw_minus1_1(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    arr = (arr / 255.0) * 2.0 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def ensure_bvt_installed(wheel_path: str = BVT_WHEEL) -> None:
    if not is_linux():
        raise RuntimeError("BVT wheel is Linux-only")
    try:
        import BVT  # noqa: F401
        return
    except ImportError:
        if not os.path.isfile(wheel_path):
            raise FileNotFoundError(f"BVT wheel not found: {wheel_path}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
```

- [ ] **Step 2: 运行单元测试**

```bash
python -m pytest tests/test_onnx_preprocess.py -v -k "image_matches or pil_to_nchw"
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/onnx_preprocess.py tests/test_onnx_preprocess.py
git commit -m "feat: add onnx preprocess pure helpers"
```

---

### Task 6: 预处理 alignment 路由

**Files:**
- Modify: `src/onnx_preprocess.py`（追加 detection + alignment + preprocess_image + preprocess_model）

- [ ] **Step 1: 实现 detection 与 alignment 路由**

在 `src/onnx_preprocess.py` 追加:

```python
from src.alignment import (
    beard_alignment,
    eyebrow_alignment,
    eyelid_alignment,
    face_alignment,
    org_alignment,
)
from src.detector.bvt_face_detector import BVTFaceDetector
from src.detector.landmarks_detector import LandmarksDetector

_bvt_detector = None
_dlib_detector = None


def _get_bvt_detector():
    global _bvt_detector
    if _bvt_detector is None:
        ensure_bvt_installed()
        _bvt_detector = BVTFaceDetector()
    return _bvt_detector


def _get_dlib_detector():
    global _dlib_detector
    if _dlib_detector is None:
        _dlib_detector = LandmarksDetector()
    return _dlib_detector


def _first_face_landmarks(img: Image.Image, use_bvt: bool):
    if use_bvt:
        faces = _get_bvt_detector().get_landmarks(img)
    else:
        faces = list(_get_dlib_detector().get_landmarks(img))
    if not faces:
        raise RuntimeError("No face detected")
    return faces[0]


def _align_eyelid_bvt(img: Image.Image, landmarks, align_size: int) -> Image.Image:
    _, left = eyelid_alignment.image_align_run(
        img, landmarks, output_size=align_size, crop_eye="left_eye", detector="bvt"
    )
    _, right = eyelid_alignment.image_align_run(
        img, landmarks, output_size=align_size, crop_eye="right_eye", detector="bvt"
    )
    merged = Image.new("RGB", (align_size, align_size))
    merged.paste(left, (0, 0))
    merged.paste(right, (0, align_size // 2))
    return merged


def align_image(img: Image.Image, align: str, align_size: int, use_bvt: bool) -> Image.Image:
    if align == "none":
        return img
    landmarks = _first_face_landmarks(img, use_bvt=use_bvt)
    detector = "bvt" if use_bvt else "dlib"

    if align == "beard":
        fn = beard_alignment if use_bvt else org_alignment
        _, aligned = fn.image_align_run(img, landmarks, output_size=align_size, detector=detector)
        return aligned
    if align == "eyebrow":
        _, aligned = eyebrow_alignment.image_align_run(
            img, landmarks, output_size=align_size, detector=detector
        )
        return aligned
    if align == "face":
        fn = face_alignment if use_bvt else org_alignment
        _, aligned = fn.image_align_run(img, landmarks, output_size=align_size, detector=detector)
        return aligned
    if align == "eyelid":
        if use_bvt:
            return _align_eyelid_bvt(img, landmarks, align_size)
        _, aligned = org_alignment.image_align_run(
            img, landmarks, output_size=align_size, detector="dlib"
        )
        return aligned
    raise ValueError(f"Unknown align type: {align}")


def preprocess_single_image(
    img: Image.Image,
    config: dict,
    preprocessed_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Returns (NCHW tensor, meta dict with keys: skipped_align, detector).
    """
    meta = {"skipped_align": False, "detector": "skipped"}
    input_size = config["input_size"]
    align = config["align"]

    if image_matches_input_size(img, input_size):
        aligned = img
        meta["skipped_align"] = True
    elif align == "none":
        aligned = img
        meta["skipped_align"] = True
    else:
        use_bvt = is_linux()
        meta["detector"] = "bvt" if use_bvt else "dlib"
        aligned = align_image(img, align, config["align_size"], use_bvt=use_bvt)

    aligned = resize_to_input_size(aligned, input_size)
    if preprocessed_path:
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        aligned.save(preprocessed_path)

    tensor = pil_to_nchw_minus1_1(aligned)
    return tensor, meta


def preprocess_model(
    config: dict,
    imgs_dir: str,
    preprocessed_dir: str,
    model_name: str,
    input_overrides: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Returns (input_name -> tensor, combined meta).
    input_overrides: optional {filename: abs_path} for CLI overrides.
    """
    tensors = {}
    meta = {"per_input": {}}
    overrides = input_overrides or {}

    for idx, (inp_name, filename) in enumerate(zip(config["input_names"], config["inputs"])):
        img_path = overrides.get(filename, os.path.join(imgs_dir, filename))
        img = Image.open(img_path).convert("RGB")
        suffix = "" if len(config["inputs"]) == 1 else f"_{inp_name}"
        pre_path = os.path.join(preprocessed_dir, f"{model_name}{suffix}.png")
        tensor, inp_meta = preprocess_single_image(img, config, pre_path)
        tensors[inp_name] = tensor
        meta["per_input"][inp_name] = {"path": img_path, **inp_meta}
    return tensors, meta
```

- [ ] **Step 2: 手动 smoke test（macOS，需 dlib）**

```bash
python -c "
from src.onnx_configs import MODEL_CONFIGS
from src.onnx_preprocess import preprocess_model
t, m = preprocess_model(MODEL_CONFIGS['gender_race'], 'imgs', 'results/_preprocessed', 'gender_race')
print(t['input'].shape, m)
"
```
Expected: `(1, 3, 128, 128)` shape，detector=skipped（22.png 非 128x128 会 resize，align=none 不检测）

- [ ] **Step 3: Commit**

```bash
git add src/onnx_preprocess.py
git commit -m "feat: add alignment routing and preprocess_model"
```

---

### Task 7: 推理、后处理与文档生成

**Files:**
- Create: `src/run_all_onnx.py`

- [ ] **Step 1: 实现完整 runner**

`src/run_all_onnx.py`:
```python
#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.onnx_configs import get_config_for_model, MODEL_CONFIGS
from src.onnx_preprocess import preprocess_model


def load_session(model_path: str, use_gpu: bool) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def postprocess_image(output: np.ndarray, channels: int) -> Image.Image:
    out = np.squeeze(output, axis=0)
    if out.ndim == 3 and out.shape[0] >= 3:
        out = out[:3] if channels >= 3 else out
    out = np.transpose(out, (1, 2, 0))
    out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def format_io_meta(session: ort.InferenceSession) -> dict:
    return {
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()],
    }


def build_preprocess_doc(config: dict, meta: dict) -> str:
    lines = [
        f"- align: {config['align']}",
        f"- input_size: {config['input_size']}",
        f"- normalize: {config['normalize']}",
    ]
    for name, inp in meta.get("per_input", {}).items():
        det = inp.get("detector", "n/a")
        skipped = inp.get("skipped_align", False)
        lines.append(f"- input `{name}`: {inp['path']} (detector={det}, skipped_align={skipped})")
    return "\n".join(lines)


def run_model(model_path, config, args, results_dir, preprocessed_dir, doc_entries):
    model_name = os.path.basename(model_path).replace(".onnx", "")
    entry = {
        "model": model_path,
        "status": "failed",
        "error": None,
        "output_path": None,
        "preprocess_meta": None,
    }
    try:
        overrides = {}
        if args.image and "22.png" in config["inputs"]:
            overrides["22.png"] = os.path.join(args.imgs_dir, args.image)
        if args.source and "0.png" in config["inputs"]:
            overrides["0.png"] = os.path.join(args.imgs_dir, args.source)

        tensors, meta = preprocess_model(config, args.imgs_dir, preprocessed_dir, model_name, overrides)
        entry["preprocess_meta"] = meta

        session = load_session(model_path, use_gpu=not args.cpu)
        entry["onnx_io"] = format_io_meta(session)

        if config["output_type"] == "classification":
            out_names = config["output_names"]
            outputs = session.run(out_names, tensors)
            result = {}
            for name, arr in zip(out_names, outputs):
                flat = np.squeeze(arr)
                result[name] = {
                    "shape": list(arr.shape),
                    "values": flat.tolist() if flat.size > 1 else float(flat),
                }
                if flat.size > 1:
                    result[name]["argmax"] = int(np.argmax(flat))
            out_path = os.path.join(results_dir, f"{model_name}.json")
            payload = {"model": os.path.basename(model_path), "inputs": meta, "outputs": result, "status": "ok"}
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            outputs = session.run(None, tensors)
            img = postprocess_image(outputs[0], config.get("output_channels", 3))
            out_path = os.path.join(results_dir, f"{model_name}.png")
            img.save(out_path)

        entry["status"] = "ok"
        entry["output_path"] = out_path
    except Exception as e:
        entry["error"] = str(e)
        traceback.print_exc()

    doc_entries.append((model_name, config, entry))
    return entry


def write_models_md(results_dir, doc_entries):
    lines = ["# ONNX Models I/O Documentation\n", f"Generated by `src/run_all_onnx.py`\n"]
    for model_name, config, entry in doc_entries:
        lines.append(f"## {model_name}\n")
        if os.path.isfile(entry["model"]):
            size_mb = os.path.getsize(entry["model"]) / (1024 * 1024)
            lines.append(f"- **File:** `{entry['model']}` ({size_mb:.2f} MB)\n")
        if entry.get("onnx_io"):
            lines.append("### ONNX I/O\n")
            for inp in entry["onnx_io"]["inputs"]:
                lines.append(f"- Input `{inp['name']}`: shape={inp['shape']}, type={inp['type']}\n")
            for out in entry["onnx_io"]["outputs"]:
                lines.append(f"- Output `{out['name']}`: shape={out['shape']}, type={out['type']}\n")
        lines.append("### Preprocessing\n")
        if entry.get("preprocess_meta"):
            lines.append(build_preprocess_doc(config, entry["preprocess_meta"]) + "\n")
        lines.append("### Run Result\n")
        lines.append(f"- Status: **{entry['status']}**\n")
        if entry.get("output_path"):
            lines.append(f"- Output: `{entry['output_path']}`\n")
        if entry.get("error"):
            lines.append(f"- Error: {entry['error']}\n")
        lines.append("\n")
    md_path = os.path.join(results_dir, "MODELS.md")
    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch ONNX inference for all models")
    parser.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models"))
    parser.add_argument("--imgs-dir", default=os.path.join(REPO_ROOT, "imgs"))
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--image", default="22.png", help="Override default 22.png")
    parser.add_argument("--source", default="0.png", help="Override face_swap source")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    preprocessed_dir = os.path.join(args.results_dir, "_preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for fn in ["22.png", "0.png"]:
        p = os.path.join(args.imgs_dir, getattr(args, fn.split(".")[0]) if fn == "22.png" else args.source if fn == "0.png" else fn)
        # validate required images exist for configured models
    for fn in set(sum([c["inputs"] for c in MODEL_CONFIGS.values()], [])):
        path = os.path.join(args.imgs_dir, args.image if fn == "22.png" else args.source if fn == "0.png" else fn)
        if not os.path.isfile(path):
            sys.exit(f"Missing input image: {path}")

    doc_entries = []
    for fname in sorted(os.listdir(args.models_dir)):
        if not fname.endswith(".onnx"):
            continue
        model_path = os.path.join(args.models_dir, fname)
        config = get_config_for_model(model_path)
        if config is None:
            print(f"WARN: no config for {fname}, skipping")
            continue
        print(f"Running {fname}...")
        run_model(model_path, config, args, args.results_dir, preprocessed_dir, doc_entries)

    write_models_md(args.results_dir, doc_entries)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 端到端运行（macOS + dlib）**

```bash
pip install -r requirements-onnx.txt
python src/run_all_onnx.py --cpu
ls results/
ls results/_preprocessed/
head -50 results/MODELS.md
```
Expected: 5 png + gender_race.json + MODELS.md + _preprocessed 中间图

- [ ] **Step 3: Commit**

```bash
git add src/run_all_onnx.py
git commit -m "feat: add batch ONNX inference runner and MODELS.md generator"
```

---

### Task 8: 更新 spec 状态与文档

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-onnx-batch-inference-design.md`

- [ ] **Step 1: 将 spec 状态改为 Approved**

```markdown
**状态:** Approved
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-onnx-batch-inference-design.md docs/superpowers/plans/
git commit -m "docs: approve spec and add implementation plan"
```

---

## Spec Coverage Checklist

| Spec Requirement | Task |
|------------------|------|
| 6 模型批量推理 | Task 7 |
| face_swap source=0.png target=22.png | Task 4 config + Task 7 overrides |
| 尺寸匹配跳过 alignment | Task 5-6 |
| Linux BVT 自动安装 | Task 5 ensure_bvt_installed |
| 非 Linux dlib 回退 | Task 6 align_image |
| vendoring 到 src/ | Task 2-3 |
| 保存 _preprocessed 中间图 | Task 6 preprocess_single_image |
| gender_race JSON | Task 7 classification branch |
| MODELS.md 文档 | Task 7 write_models_md |
| CLI 参数 | Task 7 main() |

## Manual Verification (Linux)

在 Linux cp310 机器上额外验证 BVT 路径:

```bash
pip install 3rdparty/BVT-1.48.0-cp310-cp310-linux_x86_64.whl
python src/run_all_onnx.py --cpu
grep "detector=bvt" results/MODELS.md
```
