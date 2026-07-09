"""
用 Hugging Face Hub 上的 StyleGAN（FFHQ 128×128）：`from_pretrained` 生成人脸；
支持 truncation ψ、风格混合（粗/细层）、W 空间插值、可选 W 方向偏移（属性编辑）。

关于「按性别 / 头发 / 表情指定生成」：
- 本模型为**无条件** GAN：采样 z 只能随机脸，**没有**内置类别条件输入。
- 可行路径：（1）换**带条件/文本**的生成模型（如条件 GAN、扩散 + 文本）；
  （2）**W 空间语义方向**：InterfaceGAN / GANSpace / SeFa 等在 FFHQ 的 W 上拟合超平面法向，
  存成 512 维向量；本脚本可用 `--w-direction` + `--w-direction-strength` 做 w' = w + α·d
  （方向需与当前 GAN 的 W 维一致；若换官方 FFHQ StyleGAN2，方向通常可迁移试，不保证 128 版完全一致）。
- 更强控制：StyleCLIP 文本编辑、GAN 反演后再沿方向推 latent。

macOS 优先 MPS，否则 CUDA/CPU。
依赖: pip install torch torchvision huggingface_hub safetensors numpy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image

REPO_ID = "hajar001/stylegan2-ffhq-128"
STYLEGAN_PY = "style_gan.py"


def pick_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def import_stylegan_class():
    """从 Hub 快照目录加载 `StyleGAN`（与权重同仓的 `style_gan.py`）。"""
    model_py = hf_hub_download(repo_id=REPO_ID, filename=STYLEGAN_PY)
    # 勿用 resolve()：Hub 快照里 style_gan.py 常链到 blobs/，resolve 后 parent 无同名模块
    root = str(Path(model_py).parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from style_gan import StyleGAN  # type: ignore[import-not-found]

    return StyleGAN


def to_display(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1]。"""
    x = (x + 1) * 0.5
    return torch.clamp(x, 0.0, 1.0)


def load_w_direction(path: Path, w_dim: int, device: torch.device) -> torch.Tensor:
    """载入形状 (w_dim,) 的方向向量，支持 .npy / .pt / .pth。"""
    import numpy as np

    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    elif suf in (".pt", ".pth"):
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().numpy()
        else:
            arr = np.asarray(t)
    else:
        raise ValueError(f"不支持的格式: {suf}，请用 .npy 或 .pt")
    vec = torch.as_tensor(arr, dtype=torch.float32, device=device).reshape(-1)
    if vec.numel() != w_dim:
        raise ValueError(f"方向长度 {vec.numel()} != w_dim {w_dim}")
    return vec


def w_truncated(model: torch.nn.Module, w: torch.Tensor, psi: float) -> torch.Tensor:
    if psi >= 1.0:
        return w
    return model.w_mean + psi * (w - model.w_mean)


def generate_from_z(
    model: torch.nn.Module,
    z: torch.Tensor,
    *,
    truncation: float,
    w_delta: torch.Tensor | None,
    w_delta_strength: float,
    n_layers: int,
) -> torch.Tensor:
    """z → mapping → 可选 W 偏移 → synthesis（与 `generate` 的 truncation 语义一致）。"""
    w = model.mapping(z)
    w = w_truncated(model, w, truncation)
    if w_delta is not None:
        w = w + w_delta_strength * w_delta
    w_exp = w.unsqueeze(1).expand(-1, n_layers, -1)
    return model.synthesis(w_exp)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF Hub StyleGAN2-FFHQ 128 生成与控制")
    parser.add_argument("--num-images", type=int, default=16, help="网格采样数量")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation ψ，越小越保守")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--style-mix-coarse",
        type=int,
        default=0,
        metavar="K",
        help=">0 时启用风格混合：前 K 层用 seed，其余用 seed+1",
    )
    parser.add_argument(
        "--interp-steps",
        type=int,
        default=0,
        metavar="N",
        help=">0 时在 W 空间 seed 与 seed+1 之间插 N 张图",
    )
    parser.add_argument(
        "--w-direction",
        type=str,
        default=None,
        metavar="PATH",
        help="512 维方向 .npy/.pt（如 InterfaceGAN 边界），用于语义编辑：w' = w + strength·d",
    )
    parser.add_argument(
        "--w-direction-strength",
        type=float,
        default=2.0,
        help="与 --w-direction 联用，α 越大属性越夸张，易失真",
    )
    args = parser.parse_args()

    device = pick_device()
    out_dir = Path(__file__).resolve().parent / "stylegan_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    StyleGAN = import_stylegan_class()
    model = StyleGAN.from_pretrained(REPO_ID).to(device).eval()
    n_layers = model.synthesis.num_layers

    z_dim = model.z_dim
    w_dim = model.w_dim
    # `generate` 在 ψ<1 且未缓存时会默认跑 1e4 次 mapping；预热缩短首次等待
    w_mean_samples = 2048 if args.truncation < 1.0 else 0

    w_delta: torch.Tensor | None = None
    if args.w_direction:
        w_delta = load_w_direction(Path(args.w_direction).expanduser(), w_dim, device)

    with torch.no_grad():
        if w_mean_samples:
            model.update_w_mean(num_samples=w_mean_samples)
        if args.interp_steps > 0:
            z_a = torch.randn(1, z_dim, device=device)
            z_b = torch.randn(1, z_dim, device=device)
            w_a = model.mapping(z_a)
            w_b = model.mapping(z_b)
            if args.truncation < 1.0:
                w_a = model.w_mean + args.truncation * (w_a - model.w_mean)
                w_b = model.w_mean + args.truncation * (w_b - model.w_mean)
            steps = args.interp_steps
            ws = []
            for i in range(steps):
                t = i / max(steps - 1, 1)
                w = (1.0 - t) * w_a + t * w_b
                w_exp = w.unsqueeze(1).expand(-1, n_layers, -1)
                ws.append(model.synthesis(w_exp))
            stack = torch.cat(ws, dim=0)
            path = out_dir / "w_interp.png"
            save_image(
                to_display(stack),
                path,
                nrow=min(8, steps),
            )
            print(f"W 插值 {steps} 步 → {path}")

        if args.style_mix_coarse > 0:
            k = min(max(args.style_mix_coarse, 1), n_layers - 1)
            z1 = torch.randn(1, z_dim, device=device)
            z2 = torch.randn(1, z_dim, device=device)
            w1 = model.mapping(z1)
            w2 = model.mapping(z2)
            if args.truncation < 1.0:
                w1 = model.w_mean + args.truncation * (w1 - model.w_mean)
                w2 = model.w_mean + args.truncation * (w2 - model.w_mean)
            w_mix = torch.cat(
                [
                    w1.unsqueeze(1).expand(-1, k, -1),
                    w2.unsqueeze(1).expand(-1, n_layers - k, -1),
                ],
                dim=1,
            )
            mixed = model.synthesis(w_mix)
            path = out_dir / f"style_mix_coarse{k}.png"
            save_image(to_display(mixed), path)
            print(f"风格混合（前 {k}/{n_layers} 层来自 seed={args.seed}）→ {path}")

        n = args.num_images
        z = torch.randn(n, z_dim, device=device)
        if w_delta is not None:
            images = generate_from_z(
                model,
                z,
                truncation=args.truncation,
                w_delta=w_delta,
                w_delta_strength=args.w_direction_strength,
                n_layers=n_layers,
            )
            grid_name = "grid_w_direction.png"
        else:
            images = model.generate(z, truncation_psi=args.truncation)
            grid_name = "grid.png"
        nrow = int(math.sqrt(n)) if int(math.sqrt(n)) ** 2 == n else min(8, n)
        path = out_dir / grid_name
        save_image(to_display(images), path, nrow=nrow)
        extra = f" | W 方向 α={args.w_direction_strength}" if w_delta is not None else ""
        print(f"模型: {REPO_ID} | 设备: {device} | ψ={args.truncation}{extra} | 网格 → {path}")


if __name__ == "__main__":
    main()
