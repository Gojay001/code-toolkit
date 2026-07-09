"""
LEDITS++（diffusers）：文本引导编辑；默认 SDXL（比 SD1.5 更适合作人脸局部编辑）。

本脚本：双眼皮 → 单眼皮。流程 invert → edit；`edit_threshold` 宜偏低。

默认模型：stabilityai/stable-diffusion-xl-base-1.0（LEditsPPPipelineStableDiffusionXL）。
可用 --backend sd15 回退 runwayml/stable-diffusion-v1-5。

macOS 优先 MPS；SDXL 显存占用大，默认 768×768，CUDA 可 --height 1024 --width 1024。

依赖: pip install diffusers transformers accelerate torch torchvision huggingface_hub safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import (
    DiffusionPipeline,
    LEditsPPPipelineStableDiffusion,
    LEditsPPPipelineStableDiffusionXL,
)
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "imgs" / "0.png"

MODEL_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_SD15 = "runwayml/stable-diffusion-v1-5"

DEFAULT_SOURCE_PROMPT = (
    "photorealistic portrait, person with double eyelids, visible upper eyelid crease, "
    "natural eyes, sharp facial details"
)
DEFAULT_EDIT_PROMPTS = [
    "monolid eyes",
    "single eyelids without crease",
    "smooth upper eyelid, no eyelid fold",
]
DEFAULT_NEGATIVE = (
    "double eyelids, eyelid crease, deep crease, hooded double eyelid, "
    "thick false eyelashes, blurry eyes, deformed eyes, cartoon, low quality"
)


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


def load_pipeline(backend: str, model_id: str | None, device: torch.device) -> DiffusionPipeline:
    if backend == "sdxl":
        mid = model_id or MODEL_SDXL
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        kw: dict = {"torch_dtype": dtype}
        if device.type == "cuda":
            kw["variant"] = "fp16"
        pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(mid, **kw)
    elif backend == "sd15":
        mid = model_id or MODEL_SD15
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
            mid,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        raise ValueError(f"未知 backend: {backend}，请用 sdxl 或 sd15")

    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="LEDITS++：双眼皮 → 单眼皮（默认 SDXL）")
    parser.add_argument("--input-image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument(
        "--backend",
        type=str,
        choices=("sdxl", "sd15"),
        default="sdxl",
        help="sdxl=Stable Diffusion XL（推荐）；sd15=旧版 SD1.5",
    )
    parser.add_argument("--model-id", type=str, default=None, help="覆盖默认 Hub 模型 id")
    parser.add_argument("--source-prompt", type=str, default=DEFAULT_SOURCE_PROMPT)
    parser.add_argument("--edit-prompts", type=str, nargs="+", default=DEFAULT_EDIT_PROMPTS)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE)
    parser.add_argument("--height", type=int, default=768, help="SDXL 常用 768 或 1024（须为 32 倍数）")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-inversion-steps", type=int, default=50)
    parser.add_argument("--skip", type=float, default=0.2, help="SDXL 示例常用 0.2；越小改动越强")
    parser.add_argument("--source-guidance-scale", type=float, default=3.5)
    parser.add_argument("--edit-guidance-scale", type=float, default=7.0)
    parser.add_argument("--edit-threshold", type=float, default=0.4, help="眼部局部，可试 0.32–0.5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_image).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入图不存在: {input_path}")

    out_dir = Path(__file__).resolve().parent / "ledit++_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    req = pick_device()
    pipe = load_pipeline(args.backend, args.model_id, req)
    used = try_move_pipeline(pipe, req)
    if used != req:
        print(f"警告: 设备 {req} 不可用，已退回 {used}")

    model_label = args.model_id or (MODEL_SDXL if args.backend == "sdxl" else MODEL_SD15)
    gen = torch.Generator(device=used).manual_seed(args.seed)

    print(f"backend={args.backend} | 模型={model_label} | {args.height}x{args.width}")
    print(f"反转: {input_path.name} | steps={args.num_inversion_steps} skip={args.skip}")

    inv_kw: dict = {
        "image": image,
        "source_prompt": args.source_prompt,
        "source_guidance_scale": args.source_guidance_scale,
        "num_inversion_steps": args.num_inversion_steps,
        "skip": args.skip,
        "generator": gen,
        "height": args.height,
        "width": args.width,
        "resize_mode": "fill",
    }
    if args.backend == "sdxl":
        inv_kw["negative_prompt"] = args.negative_prompt

    inv_out = pipe.invert(**inv_kw)
    recon_path = out_dir / "invert_recon.png"
    inv_out.images[0].save(recon_path)
    print(f"反转重建预览 → {recon_path}")

    n_edits = len(args.edit_prompts)
    out = pipe(
        editing_prompt=args.edit_prompts,
        reverse_editing_direction=[False] * n_edits,
        edit_guidance_scale=[args.edit_guidance_scale] * n_edits,
        edit_threshold=[args.edit_threshold] * n_edits,
        negative_prompt=args.negative_prompt,
        generator=gen,
    )
    edited = out.images[0]
    out_path = out_dir / f"monolid_edit_{args.backend}.png"
    edited.save(out_path)
    print(f"设备: {used} | 单眼皮编辑 → {out_path}")


if __name__ == "__main__":
    main()
