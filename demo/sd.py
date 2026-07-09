"""
Hugging Face diffusers：Stable Diffusion 1.5 文生图 / 图生图 / 可选 IP-Adapter 参考图。

- 文本：`--prompt`、`--negative-prompt`、`--guidance-scale`
- 图像结构：`--init-image` + `--strength`（img2img，越接近 1 越不保留原构图）
- 参考图语义（不强制 init）：`--ip-adapter-image`（自动加载 h94/IP-Adapter SD1.5 权重）

默认 `runwayml/stable-diffusion-v1-5`。macOS 优先 MPS，否则 CUDA/CPU。

依赖: pip install diffusers transformers accelerate torch torchvision huggingface_hub safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image


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


def maybe_ip_adapter(pipe: StableDiffusionPipeline, ip_path: Path | None, scale: float) -> None:
    if ip_path is None:
        return
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.safetensors",
    )
    pipe.set_ip_adapter_scale(scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF diffusers Stable Diffusion：文本 + 可选图像控制")
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hub 上的 SD1.5 模型 id",
    )
    parser.add_argument("--prompt", type=str, default="a photo of a woman face with bald head (no hair)")
    parser.add_argument("--negative-prompt", type=str, default="low quality, blurry, watermark, text")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-image",
        type=str,
        default=None,
        help="参考构图的 RGB 图路径；设则走 img2img",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.65,
        help="img2img 加噪强度，(0,1]，越大越偏离原图",
    )
    parser.add_argument(
        "--ip-adapter-image",
        type=str,
        default=None,
        help="IP-Adapter 参考图路径（可与 init-image 同时用）",
    )
    parser.add_argument(
        "--ip-adapter-scale",
        type=float,
        default=0.6,
        help="IP-Adapter 影响强度",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "sd_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    req = pick_device()
    dtype = torch.float16 if req.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    ip_path = Path(args.ip_adapter_image).expanduser() if args.ip_adapter_image else None
    if ip_path is not None and not ip_path.is_file():
        raise FileNotFoundError(f"IP-Adapter 参考图不存在: {ip_path}")
    maybe_ip_adapter(pipe, ip_path, args.ip_adapter_scale)

    used = try_move_pipeline(pipe, req)
    if used != req:
        print(f"警告: 设备 {req} 不可用，已退回 {used}")
    if used.type != "cuda" and dtype == torch.float16:
        pipe.to(dtype=torch.float32)

    gen = torch.Generator(device=used).manual_seed(args.seed)
    ip_pil = Image.open(ip_path).convert("RGB") if ip_path else None

    kwargs_ip = {}
    if ip_pil is not None:
        kwargs_ip["ip_adapter_image"] = ip_pil

    init_path = Path(args.init_image).expanduser() if args.init_image else None
    if init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"init-image 不存在: {init_path}")

    if init_path is not None:
        init_pil = Image.open(init_path).convert("RGB")
        init_pil = init_pil.resize((args.width, args.height), Image.Resampling.LANCZOS)
        i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
        i2i = i2i.to(used)
        out = i2i(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=init_pil,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            **kwargs_ip,
        )
        suffix = "img2img"
    else:
        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            **kwargs_ip,
        )
        suffix = "txt2img"

    img = out.images[0]
    out_path = out_dir / f"sd_{suffix}.png"
    img.save(out_path)
    print(f"模型: {args.model_id} | 设备: {used} | 模式: {suffix} | 输出: {out_path}")


if __name__ == "__main__":
    main()
