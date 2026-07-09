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
