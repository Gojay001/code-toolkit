import os
import cv2
from PIL import Image
import numpy as np
import onnxruntime as ort


def load_onnx_model(model_path, use_gpu=True):
    """
    加载ONNX模型

    Args:
        model_path (str): ONNX模型文件路径
        use_gpu (bool): 是否使用GPU加速（默认True）

    Returns:
        ort.InferenceSession: ONNX推理会话
    """
    options = ort.SessionOptions()

    # 配置执行提供者
    if use_gpu:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    else:
        providers = ['CPUExecutionProvider']

    session = ort.InferenceSession(model_path, options, providers=providers)

    print(f"模型加载成功: {model_path}")
    print(f"使用的执行提供者: {session.get_providers()}")

    return session


def preprocess_image(image_path, target_size=(256, 256), normalize=True):
    """
    预处理图像为ONNX模型输入格式

    Args:
        image_path (str): 图像文件路径
        target_size (tuple): 目标尺寸 (width, height)
        normalize (bool): 是否归一化到[-1, 1]

    Returns:
        numpy.ndarray: 预处理的图像数组，形状为 (1, C, H, W)，dtype为float32
    """
    try:
        # 读取图像
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)  # 调整尺寸
        img_array = np.array(img).astype(np.float32)

        if normalize:
            # 归一化到[-1, 1]
            img_array = (img_array / 255.0) * 2.0 - 1.0
        else:
            # 归一化到[0, 1]
            img_array = img_array / 255.0

        # 转换为CHW格式 (Channels, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))

        # 添加batch维度 (B, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except FileNotFoundError:
        print(f"错误: 图像文件未找到 {image_path}")
        return None
    except Exception as e:
        print(f"图像处理错误: {e}")
        return None


def postprocess_image(output, output_path=None):
    """
    后处理模型输出，转换为可保存的图像

    Args:
        output (numpy.ndarray): 模型输出
        output_path (str): 保存路径（可选）

    Returns:
        PIL.Image: 处理后的图像
    """
    # 移除batch维度
    output_image = np.squeeze(output, axis=0)

    # 如果输出包含多个通道，只取前3个通道（RGB）
    if len(output_image.shape) == 3 and output_image.shape[0] > 3:
        output_image = output_image[:3, :, :]

    # 转换为HWC格式 (Height, Width, Channels)
    output_image = np.transpose(output_image, (1, 2, 0))

    # 从[-1, 1]范围转换回[0, 255]
    output_image = np.clip((output_image + 1.0) * 127.5, 0, 255).astype(np.uint8)

    # 转换为PIL图像
    output_img = Image.fromarray(output_image)

    if output_path:
        output_img.save(output_path)
        print(f"结果已保存到: {output_path}")

    return output_img


def run_onnx_inference(session, input_dict):
    """
    运行ONNX模型推理

    Args:
        session (ort.InferenceSession): ONNX会话
        input_dict (dict): 输入字典，键为输入名称，值为输入数据

    Returns:
        list: 模型输出列表
    """
    outputs = session.run(None, input_dict)
    return outputs


def inference_onnx(model_path, source_img_path, target_img_path=None, output_path=None):
    """
    完整的ONNX推理流程

    Args:
        model_path (str): ONNX模型路径
        source_img_path (str): 源图像路径
        target_img_path (str): 目标图像路径（可选）
        output_path (str): 输出图像路径（可选）

    Returns:
        numpy.ndarray: 模型输出
    """
    # 加载模型
    session = load_onnx_model(model_path)

    # 获取模型输入信息
    input_names = [input.name for input in session.get_inputs()]
    print(f"模型输入: {input_names}")

    # 预处理图像
    source_img = preprocess_image(source_img_path)
    if source_img is None:
        return None

    # 构建输入字典
    input_dict = {input_names[0]: source_img}

    # 如果有第二个输入（如目标图像）
    if len(input_names) > 1 and target_img_path is not None:
        target_img = preprocess_image(target_img_path)
        if target_img is not None:
            input_dict[input_names[1]] = target_img

    # 运行推理
    outputs = run_onnx_inference(session, input_dict)
    print(f"输出形状: {outputs[0].shape}")

    # 后处理并保存结果
    if output_path:
        postprocess_image(outputs[0], output_path)
    else:
        postprocess_image(outputs[0])

    return outputs[0]


if __name__ == "__main__":
    # 示例用法
    model_path = "/Users/bigo10295/Downloads/generator_lapa_ffhq_style_1_align_aug_512_lpsnet_v2_best_model_wrapper_v2.onnx"
    source_img_path = '/Users/bigo10295/Downloads/test_data/source_images_aligned/dilireba.jpg'
    # target_img_path = '/Users/bigo10295/Downloads/temp_img/yangmi_256x256.jpg'
    output_path = '/Users/bigo10295/Downloads/test_data/output.png'

    # 运行推理
    result = inference_onnx(
        model_path=model_path,
        source_img_path=source_img_path,
        # target_img_path=target_img_path,
        output_path=output_path
    )

    print("推理完成！")

