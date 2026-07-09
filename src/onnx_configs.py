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
