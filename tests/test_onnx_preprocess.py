import numpy as np
from PIL import Image

from src.onnx_configs import get_config_for_model, MODEL_CONFIGS


def test_image_matches_input_size_exact():
    from src.onnx_preprocess import image_matches_input_size

    img = Image.new("RGB", (256, 192))  # W=256, H=192
    assert image_matches_input_size(img, (192, 256)) is True


def test_image_matches_input_size_mismatch():
    from src.onnx_preprocess import image_matches_input_size

    img = Image.new("RGB", (512, 512))
    assert image_matches_input_size(img, (192, 256)) is False


def test_pil_to_nchw_minus1_1_shape_and_range():
    from src.onnx_preprocess import pil_to_nchw_minus1_1

    img = Image.new("RGB", (128, 128), color=(255, 0, 0))
    arr = pil_to_nchw_minus1_1(img)
    assert arr.shape == (1, 3, 128, 128)
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0 and arr.min() >= -1.0


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
