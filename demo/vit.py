"""
使用 Hugging Face / timm 的 ViT：读取图像、提取 CLS 特征并做 ImageNet 分类。
macOS 上优先使用 MPS（Apple GPU），否则 CPU。
依赖: pip install timm torch torchvision pillow
"""

from __future__ import annotations

import urllib.request

import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# timm 内置 ViT，权重通常来自 Hugging Face Hub（pretrained=True）
MODEL_NAME = "vit_base_patch16_224"
IMAGE_PATH = "/Users/bigo10295/Downloads/code-toolkit/imgs/0.png"
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)


def load_imagenet_labels() -> list[str]:
    try:
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def extract_cls_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ViT: forward_features 为 patch 序列，取 CLS token（索引 0）作为全局特征。"""
    feat = model.forward_features(x)
    if feat.dim() == 3:
        return feat[:, 0]
    return feat


def pick_device() -> torch.device:
    """macOS：优先 Apple GPU（MPS），否则 CPU。"""
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    labels = load_imagenet_labels()

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval()
    model.to(device)

    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)

    img = Image.open(IMAGE_PATH).convert("RGB")
    img = img.resize((224, 224))
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top5_p, top5_i = torch.topk(probs, min(5, probs.shape[1]), dim=1)
        feat = extract_cls_features(model, x)

    print(f"模型: {MODEL_NAME} (timm {timm.__version__})")
    print(f"设备: {device}")
    print(f"图像: {IMAGE_PATH}")
    print(f"特征向量 shape: {tuple(feat.shape)} (CLS)，L2 范数: {feat.norm(dim=-1).item():.4f}")
    print("Top-5 分类:")
    for rank in range(top5_i.shape[1]):
        idx = int(top5_i[0, rank].item())
        name = labels[idx] if idx < len(labels) else str(idx)
        print(f"  {rank + 1}. {name}  p={top5_p[0, rank].item():.4f}")


if __name__ == "__main__":
    main()
