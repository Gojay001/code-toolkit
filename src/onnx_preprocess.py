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
