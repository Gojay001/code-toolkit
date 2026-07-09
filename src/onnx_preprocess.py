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
