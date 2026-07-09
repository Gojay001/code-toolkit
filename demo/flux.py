"""
FLUX.2 Klein Base：按指令编辑人脸图（默认去除眉毛）。

输入默认 imgs/0.png，用 image + prompt 做编辑（非纯文生图）。
模型 black-forest-labs/FLUX.2-klein-base-4B（支持多参考/图编辑）。

macOS 用 MPS + CPU offload；需较大内存（约 16GB+ 统一内存更稳）。

依赖: pip install diffusers transformers accelerate torch huggingface_hub safetensors
Hub: export HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "imgs" / "0.png"
DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"

DEFAULT_PROMPT = (
    "Edit this portrait photo: remove the eyebrows completely, smooth natural skin on the brow area, "
    "keep the same person, same eyes, nose, mouth, hair, skin tone and lighting, "
    "photorealistic, unchanged background"
)
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="FLUX.2 Klein：编辑人脸图（默认去眉毛）")
    parser.add_argument("--input-image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Base 模型建议约 50 步")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_image).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入图不存在: {input_path}")

    out_dir = Path(__file__).resolve().parent / "flux_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "no_eyebrows.png"

    device = pick_device()
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

    source = load_image(input_path)
    print(f"加载模型: {args.model_id} | 设备: {device} | 输入: {input_path}")

    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()

    gen_device = device.type if device.type in ("cuda", "mps") else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        image=source,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=gen,
    ).images[0]
    result.save(out_path)
    print(f"已保存 → {out_path}")


if __name__ == "__main__":
    main()
