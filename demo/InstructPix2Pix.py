"""
Hugging Face diffusers：InstructPix2Pix 按自然语言指令编辑图像。

默认模型 timbrooks/instruct-pix2pix，默认输入仓库 imgs/0.png。
macOS 优先 MPS，否则 CUDA/CPU；非 CUDA 使用 float32。

依赖: pip install diffusers transformers accelerate torch torchvision pillow requests huggingface_hub
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "imgs" / "0.png"
DEFAULT_MODEL = "timbrooks/instruct-pix2pix"
DEFAULT_PROMPT = "make the person have monolid eyes, single eyelids, no upper eyelid crease"


def pick_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def try_move_pipeline(pipe: StableDiffusionInstructPix2PixPipeline, device: torch.device) -> torch.device:
    if device.type == "cpu":
        pipe.to(device)
        return device
    try:
        pipe.to(device)
        return device
    except Exception:
        pipe.to(torch.device("cpu"))
        return torch.device("cpu")


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="InstructPix2Pix 指令式图像编辑")
    parser.add_argument("--input-image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="文本引导强度",
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="保留原图程度；越大越贴近输入，过小易偏离构图",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_image).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入图不存在: {input_path}")

    out_dir = Path(__file__).resolve().parent / "instructpix2pix_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    req = pick_device()
    dtype = torch.float16 if req.type == "cuda" else torch.float32

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    used = try_move_pipeline(pipe, req)
    if used != req:
        print(f"警告: 设备 {req} 不可用，已退回 {used}")
    if used.type != "cuda" and dtype == torch.float16:
        pipe.to(dtype=torch.float32)

    gen = torch.Generator(device=used).manual_seed(args.seed)

    out = pipe(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=gen,
    )
    out_path = out_dir / "edit.png"
    out.images[0].save(out_path)
    print(
        f"模型: {args.model_id} | 设备: {used} | "
        f"guidance={args.guidance_scale} image_guidance={args.image_guidance_scale} | "
        f"输出: {out_path}"
    )


if __name__ == "__main__":
    main()
