"""
人脸属性 ONNX 推理脚本
输入: input
输出: output_gender, output_race
"""

import os
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort


INPUT_NAME = "input"
OUTPUT_NAMES = ["output_gender", "output_race"]


def load_onnx_model(model_path, use_gpu=True):
    """
    加载 ONNX 模型

    Args:
        model_path (str): ONNX 模型文件路径
        use_gpu (bool): 是否使用 GPU（默认 True）

    Returns:
        ort.InferenceSession: ONNX 推理会话
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        model_path,
        sess_options=ort.SessionOptions(),
        providers=providers,
    )
    print(f"模型加载成功: {model_path}")
    print(f"执行提供者: {session.get_providers()}")
    return session


def preprocess_image(image_path, target_size=(128, 128), normalize=True):
    """
    预处理图像为模型输入格式

    Args:
        image_path (str): 图像路径
        target_size (tuple): (H, W)
        normalize (bool): 是否归一化到 [0, 1] 或标准化

    Returns:
        np.ndarray: (1, C, H, W), float32
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # (W, H) for PIL
    arr = np.array(img).astype(np.float32)

    if normalize:
        arr = arr / 127.5 - 1.0
    # NCHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def _get_model_input_hw(session):
    """
    从 ONNX 模型的第一个输入推断 (H, W)，若为动态则回退到 (128, 128)
    """
    try:
        input_meta = session.get_inputs()[0]
        shape = list(input_meta.shape)
        if len(shape) >= 4:
            h, w = shape[-2], shape[-1]
            if isinstance(h, int) and isinstance(w, int):
                return int(h), int(w)
    except Exception:
        pass
    return 128, 128


def run_face_attr(session, image_path, input_size=None):
    """
    运行人脸属性推理

    Args:
        session: ONNX InferenceSession
        image_path (str): 输入图像路径
        input_size (tuple | None): 模型输入 (H, W)，若为 None 则自动从模型推断

    Returns:
        dict: {"output_gender": np.ndarray, "output_race": np.ndarray}
    """
    if input_size is None:
        input_size = _get_model_input_hw(session)

    input_tensor = preprocess_image(image_path, target_size=input_size)
    inputs = {INPUT_NAME: input_tensor}

    # 按模型定义的输出名获取结果
    outputs = session.run(OUTPUT_NAMES, inputs)

    return {
        OUTPUT_NAMES[0]: outputs[0],
        OUTPUT_NAMES[1]: outputs[1],
    }


def parse_attr_output(output_gender, output_race, top_k=5):
    """
    解析属性输出（假设为 logits/概率）
    output_gender: (1, n_gender) -> 取 argmax 或 softmax 后解释
    output_race:  (1, n_race)  -> 同上
    """
    gender = np.squeeze(output_gender)
    race = np.squeeze(output_race)

    # 若为 logits，转成概率
    if gender.ndim >= 1:
        exp_g = np.exp(gender - gender.max())
        gender_prob = exp_g / exp_g.sum()
    else:
        gender_prob = np.array([gender])
    if race.ndim >= 1:
        exp_r = np.exp(race - race.max())
        race_prob = exp_r / exp_r.sum()
    else:
        race_prob = np.array([race])

    gender_idx = int(np.argmax(gender_prob))
    race_idx = int(np.argmax(race_prob))

    return {
        "gender": {"index": gender_idx, "prob": float(gender_prob[gender_idx])},
        "race": {"index": race_idx, "prob": float(race_prob[race_idx])},
        "gender_all": gender_prob,
        "race_all": race_prob,
    }


def main():
    parser = argparse.ArgumentParser(description="人脸属性 ONNX 推理 (output_gender, output_race)")
    parser.add_argument("--model", "-m", required=True, help="ONNX 模型路径")
    parser.add_argument("--image", "-i", required=True, help="输入人脸图像路径")
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="输入尺寸 (H W)，默认从模型输入自动推断",
    )
    parser.add_argument("--cpu", action="store_true", help="仅使用 CPU")
    args = parser.parse_args()

    session = load_onnx_model(args.model, use_gpu=not args.cpu)
    size = tuple(args.size) if args.size is not None else None
    result = run_face_attr(session, args.image, input_size=size)
    print("output_gender shape:", result["output_gender"].shape)
    print("output_race shape:", result["output_race"].shape)
    print(f"result: {result}")


if __name__ == "__main__":

    model_path = "../models/MobileNet_v2_n8k5_gender_beauty_fast_model_ckpt.onnx"
    image_path = "../imgs/0000001_rectalign.jpg"

    session = load_onnx_model(model_path, use_gpu=False)
    result = run_face_attr(session, image_path, input_size=None)
    print(f"result: {result}")
