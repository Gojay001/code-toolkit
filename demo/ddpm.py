"""
用 Hugging Face diffusers 的类条件扩散：DiTPipeline（ImageNet 类条件），通过 `class_labels` 生成指定类别 256×256 图像。

说明：原版 unconditional `DDPMPipeline`（如 google/ddpm-cifar10-32）无类别输入；若需「按类生成」，应使用带 class embedding 的模型。
DiT 为 Transformer 骨干的扩散模型，非 vanilla U-Net DDPM，但满足「class_labels 控语义类」这一需求。

默认 `facebook/DiT-XL-2-256`（权重较大，首次下载久）；可用 `--model-id` 换其它 DiT 仓库。
类 id 为 ImageNet-1k 索引 0–999（1000 类在部分实现中作 null 用于 CFG，勿用作生成目标）。

macOS 优先 MPS，否则 CUDA/CPU；MPS/CUDA 不可用时回退 CPU。

依赖: pip install diffusers torch torchvision accelerate huggingface_hub
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image


def pick_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def try_move_pipeline(pipe: DiffusionPipeline, device: torch.device) -> torch.device:
    if device.type == "cpu":
        pipe.to(device)
        return device
    try:
        pipe.to(device)
        return device
    except Exception:
        pipe.to(torch.device("cpu"))
        return torch.device("cpu")


def parse_class_labels(s: str) -> list[int]:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("class_labels 为空")
    ids = [int(p) for p in parts]
    for i in ids:
        if i < 0 or i > 999:
            raise ValueError(f"ImageNet 类 id 须在 0–999 内，收到 {i}")
    return ids


def expand_labels(ids: list[int], batch_size: int) -> list[int]:
    if len(ids) == batch_size:
        return ids
    if len(ids) == 1:
        return ids * batch_size
    if len(ids) < batch_size:
        return [ids[i % len(ids)] for i in range(batch_size)]
    raise ValueError(
        f"class_labels 数量 {len(ids)} 大于 batch_size {batch_size}；请减少 id 或增大 batch_size。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HF diffusers 类条件扩散（DiT）按 class_labels 生成图像")
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/DiT-XL-2-256",
        help="Hub 上的 DiTPipeline 模型 id",
    )
    parser.add_argument(
        "--class-labels",
        type=str,
        default="207",
        help="ImageNet 类 id，逗号分隔；仅 1 个时自动重复到 batch_size",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="ImageNet 类名字符串，用 | 分隔（如 'golden retriever|tabby'）；加载后解析，覆盖 --class-labels",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="生成张数，与 class_labels 展开长度一致")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="去噪步数；已换 DPM-Solver++，通常 20–50 可用",
    )
    parser.add_argument("--guidance-scale", type=float, default=4.0, help=">1 启用 classifier-free guidance")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "ddpm_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    req = pick_device()
    dtype = torch.float16 if req.type == "cuda" else torch.float32

    pipe = DiTPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    used = try_move_pipeline(pipe, req)
    if used != req:
        print(f"警告: 设备 {req} 不可用，已退回 {used}")
    if used.type != "cuda" and dtype == torch.float16:
        pipe = pipe.to(dtype=torch.float32)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.class_names:
        words = [w.strip() for w in args.class_names.split("|") if w.strip()]
        class_ids = pipe.get_label_ids(words)
        class_ids = expand_labels(class_ids, args.batch_size)
    else:
        class_ids = expand_labels(parse_class_labels(args.class_labels), args.batch_size)

    gen = torch.Generator(device=used).manual_seed(args.seed)

    out = pipe(
        class_labels=class_ids,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=gen,
    )
    images = out.images

    tensors = torch.stack([pil_to_tensor(im).float() / 255.0 for im in images])
    nrow = int(math.sqrt(len(images))) if int(math.sqrt(len(images))) ** 2 == len(images) else min(4, len(images))
    path = out_dir / "dit_class_cond_grid.png"
    save_image(tensors, path, nrow=nrow)

    print(
        f"模型: {args.model_id} | 设备: {used} | steps={args.num_inference_steps} | "
        f"cfg={args.guidance_scale} | classes={class_ids} | 输出: {path}"
    )


if __name__ == "__main__":
    main()
