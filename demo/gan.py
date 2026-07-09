"""
用 Hugging Face datasets 加载 MNIST，PyTorch 实现 DCGAN 训练并生成 28×28 手写体图像。
macOS 优先 MPS，否则 CPU。
依赖: pip install torch torchvision datasets pillow
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


def pick_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Generator(nn.Module):
    """DCGAN 生成器：z → 1×28×28（空间从 7×7 上采样到 28×28）。"""

    def __init__(self, nz: int = 100, ngf: int = 64) -> None:
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.fc = nn.Linear(nz, ngf * 8 * 7 * 7)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        x = self.fc(z).view(-1, self.ngf * 8, 7, 7)
        return self.conv(x)


class Discriminator(nn.Module):
    """DCGAN 判别器：1×28×28 → 真假 logit。"""

    def __init__(self, ndf: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(ndf * 4 * 3 * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x).flatten(1)
        return self.fc(y).squeeze(1)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF datasets + DCGAN on MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = pick_device()
    nz = 100
    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.0002
    beta1 = 0.5
    out_dir = Path(__file__).resolve().parent / "gan_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face datasets：MNIST
    ds = load_dataset("mnist", split="train")
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    class HfMnist(torch.utils.data.Dataset):
        def __init__(self, hf_split) -> None:
            self.hf_split = hf_split

        def __len__(self) -> int:
            return len(self.hf_split)

        def __getitem__(self, idx: int):
            row = self.hf_split[idx]
            img = row["image"].convert("L")
            return tfm(img)

    loader = DataLoader(
        HfMnist(ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    G = Generator(nz=nz).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(epochs):
        for i, real_cpu in enumerate(loader):
            real = real_cpu.to(device)
            bsz = real.size(0)
            label_real = torch.full((bsz,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((bsz,), fake_label, dtype=torch.float, device=device)

            # 更新 D
            D.zero_grad()
            out_real = D(real)
            loss_d_real = criterion(out_real, label_real)
            noise = torch.randn(bsz, nz, 1, 1, device=device)
            fake = G(noise).detach()
            out_fake = D(fake)
            loss_d_fake = criterion(out_fake, label_fake)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # 更新 G
            G.zero_grad()
            noise = torch.randn(bsz, nz, 1, 1, device=device)
            fake = G(noise)
            out = D(fake)
            loss_g = criterion(out, label_real)
            loss_g.backward()
            opt_g.step()

            if i % 200 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {i}/{len(loader)} "
                    f"loss_d {loss_d.item():.4f} loss_g {loss_g.item():.4f}"
                )

        with torch.no_grad():
            fakes = G(fixed_noise)
        grid_path = out_dir / f"epoch_{epoch + 1:02d}.png"
        save_image(
            fakes,
            grid_path,
            nrow=int(math.sqrt(fakes.size(0))),
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"已保存: {grid_path}")

    # 额外保存一张 8×8 拼图为 PIL 友好查看
    with torch.no_grad():
        samples = G(fixed_noise).cpu()
    save_image(
        samples,
        out_dir / "gan_samples.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"设备: {device} | DCGAN + datasets mnist | 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
