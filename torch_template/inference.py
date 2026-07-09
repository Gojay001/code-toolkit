import os

import cv2
from PIL import Image
import numpy as np

import coremltools as ct
import onnxruntime as ort

base_dir = '/Users/bigo10295/Downloads/'

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Loads, preprocesses, and converts an image to a NumPy array suitable for ONNX input.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The desired (width, height) of the image.

    Returns:
        numpy.ndarray: A NumPy array representing the preprocessed image,
                       with shape (1, C, H, W) and dtype float32.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)  # Resize the image
        img_array = np.array(img).astype(np.float32)

        # Normalize the image (optional, but often improves results)
        img_array = (img_array / 255.0) * 2.0 - 1.0

        # Transpose the image to CHW format (Channels, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add a batch dimension (B, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def inference_coreml():
    model = ct.models.MLModel("/Users/bigo10295/Downloads/gan_models/faceswap/0915/faceswap_fp32_-1_1.mlpackage")

    source_img_path = base_dir + 'temp_img/liuyifei_256x256.jpg'
    target_img_path = base_dir + 'temp_img/yangmi_256x256.jpg'
    source_img = preprocess_image(source_img_path)
    target_img = preprocess_image(target_img_path)

    predictions = model.predict({
        "source": source_img,
        "target": target_img
    })

    print("Output shape:", predictions["output"].shape)
    output_image = predictions["output"]  # Assuming the first output is the image
    output_image = np.squeeze(output_image, axis=0)  # Remove batch dimension
    output_image = output_image[:3,:,:]
    output_image = np.transpose(output_image, (1, 2, 0))  # CHW to HWC
    output_image = np.clip(output_image * 127.5 + 127.5, 0, 255).astype(np.uint8)  # Scale and convert to uint8

    output_img = Image.fromarray(output_image)
    output_img.save(base_dir + "temp_img/output.png")
    print("Inference successful. Output saved to output.png")

#--------------------------------------------------------------

def load_onnx_model(model_path):
    options = ort.SessionOptions()

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

    return ort.InferenceSession(model_path, options, providers=providers)

def run_onnx_model(session, input):
    outputs = session.run(None, input)
    return outputs[0]

def inference_onnx():
    model_path = "/Users/bigo10295/Downloads/gan_models/faceswap/0915/faceswap_fp32_-1_1.onnx"
    session = load_onnx_model(model_path)

    source_img_path = base_dir + 'temp_img/liuyifei_256x256.jpg'
    target_img_path = base_dir + 'temp_img/yangmi_256x256.jpg'
    source_img = preprocess_image(source_img_path)
    target_img = preprocess_image(target_img_path)

    # source_img_py = np.load(base_dir + 'test_data/0915/source_img_py.npy')
    # np.savetxt('source_img_py.txt', source_img_py.flatten(), fmt='%.12f')
    # np.savetxt('source_img_py_flat.txt', source_img_py.flatten(), fmt='%.12f')
    # source_close = np.allclose(source_img, source_img_py, atol=1e-6)
    # source_equal = np.array_equal(source_img, source_img_py)
    # print("Source images close:", source_close)
    # print("Source images equal:", source_equal)
    # target_img_py = np.load(base_dir + 'test_data/0915/target_img_py.npy')
    # np.savetxt('target_img_py.txt', target_img_py.flatten(), fmt='%.12f')
    # np.savetxt('target_img_py_flat.txt', target_img_py.flatten(), fmt='%.12f')
    # target_close = np.allclose(target_img, target_img_py, atol=1e-6)
    # target_equal = np.array_equal(target_img, target_img_py)
    # print("Target images close:", target_close)
    # print("Target images equal:", target_equal)

    input_data = {
        "source": source_img,
        "target": target_img
    }

    output = run_onnx_model(session, input_data)
    print("Output shape:", output.shape)

    output_image = output  # Assuming the first output is the image
    output_image = np.squeeze(output_image, axis=0)  # Remove batch dimension
    output_image = output_image[:3,:,:]
    output_image = np.transpose(output_image, (1, 2, 0))  # CHW to HWC
    output_image = np.clip(output_image * 127.5 + 127.5, 0, 255).astype(np.uint8)  # Scale and convert to uint8

    py_out_image = cv2.imread(base_dir + 'test_data/0915/py.png')
    py_out_image = cv2.cvtColor(py_out_image, cv2.COLOR_BGR2RGB)
    np.savetxt('output_image.txt', output_image.flatten(), fmt='%d')
    np.savetxt('py_out_image_flat.txt', py_out_image.flatten(), fmt='%d')
    close = np.allclose(output_image, py_out_image, atol=1e-6)
    print("Output images close:", close)

    output_img = Image.fromarray(output_image)
    output_img.save(base_dir + "temp_img/output.png")
    print("Inference successful. Output saved to output.png")

if __name__ == "__main__":
    # inference_coreml()
    inference_onnx()

    # img_py = cv2.imread('/Users/bigo10295/Downloads/test_data/0915/py.png')
    # img_onnx = cv2.imread('/Users/bigo10295/Downloads/test_data/0915/output.png')

    # diff = np.abs(img_py.astype(np.int32) - img_onnx.astype(np.int32))
    # print("Max difference:", np.max(diff))
    # print("Mean difference:", np.mean(diff))