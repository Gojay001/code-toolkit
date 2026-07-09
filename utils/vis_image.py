import os
import argparse
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


LABELS = ("1_2", "2_2")
COLS = 10
PER_LABEL = 5
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_TILE = (540, 960)  # (W, H)


def _list_images(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in EXTS:
            files.append(p)
    files.sort()
    return files


def _read_rgb(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    读取并转换为 RGB ndarray。

    - 统一输出 size（默认 TARGET_TILE）。
    - 若原图 w==h（正方形），先上下补白使其接近 9:16，再缩放到目标尺寸。
    - 非正方形图像直接缩放到目标尺寸（允许形变）。
    """
    if size is None:
        size = TARGET_TILE

    img = Image.open(path).convert("RGB")
    w, h = img.size

    if w == h:
        target_w, target_h = size
        # 目标纵横比 h/w = target_h/target_w
        desired_h = int(np.ceil(w * (target_h / target_w)))
        if desired_h < h:
            desired_h = h
        padded = Image.new("RGB", (w, desired_h), (255, 255, 255))
        y0 = (desired_h - h) // 2
        padded.paste(img, (0, y0))
        img = padded

    img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _blank_rgb(size: Tuple[int, int], color=(255, 255, 255)) -> np.ndarray:
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = np.array(color, dtype=np.uint8)
    return arr


def build_rows(root: str, tile_size: Tuple[int, int]) -> List[Tuple[str, List[Tuple[np.ndarray, str, str]]]]:
    """
    返回二维列表 rows[row_idx][col_idx] -> RGB ndarray
    每个 id 对应一行，共 6 列：
      - col 0..2: label=1_2 的前 3 张（不足补白）
      - col 3..5: label=2_2 的前 3 张（不足补白）
    """
    ids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    ids.sort()

    rows: List[Tuple[str, List[Tuple[np.ndarray, str, str]]]] = []
    for id_name in ids:
        # 读取两个 label 下的全部图片，合并后按文件名排序
        merged: List[Tuple[str, str]] = []  # (path, label)
        for label in LABELS:
            label_dir = os.path.join(root, id_name, label)
            for p in _list_images(label_dir):
                merged.append((p, label))

        merged.sort(key=lambda x: os.path.basename(x[0]))
        merged = merged[:COLS]  # 期望总共 10 张（COLS=10）

        row_tiles: List[Tuple[np.ndarray, str, str]] = []
        for p, label in merged:
            row_tiles.append((_read_rgb(p, size=tile_size), label, os.path.basename(p)))

        # 不足 COLS 的补白
        while len(row_tiles) < COLS:
            row_tiles.append((_blank_rgb(tile_size), "", ""))

        rows.append((id_name, row_tiles[:COLS]))
    return rows


def render_grid(
    rows: List[Tuple[str, List[Tuple[np.ndarray, str, str]]]],
    out_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
    force_nrows: Optional[int] = None,
):
    if not rows:
        print("未找到任何图片可展示。")
        return

    nrows = force_nrows if force_nrows is not None else len(rows)
    ncols = COLS

    fig_w = ncols * 2.0
    fig_h = nrows * 2.6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))

    # axes 在 nrows=1 时不是二维
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if r < len(rows):
                row_id, row_tiles = rows[r]
                tile, tile_label, tile_name = row_tiles[c]
            else:
                row_id, tile, tile_label, tile_name = "", _blank_rgb((TARGET_TILE[0], TARGET_TILE[1])), "", ""

            ax.imshow(tile)
            # 在每行第一张图上写 id
            if c == 0 and row_id:
                ax.text(
                    0.01,
                    0.99,
                    row_id,
                    color="red",
                    fontsize=11,
                    weight="bold",
                    va="top",
                    ha="left",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                )
            # 每张图都写 label
            if tile_label:
                ax.text(
                    0.01,
                    0.90,
                    tile_label,
                    color="blue",
                    fontsize=10,
                    weight="bold",
                    va="top",
                    ha="left",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                )
            # 每张图都写文件名
            if tile_name:
                ax.text(
                    0.01,
                    0.82,
                    tile_name,
                    color="black",
                    fontsize=9,
                    va="top",
                    ha="left",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                )
            ax.axis("off")

    plt.tight_layout(pad=0.2)
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"已保存: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="按 id 行展示图片：每行 10 张（1_2 前5张 + 2_2 前5张，不足补白）"
    )
    parser.add_argument(
        "--path",
        "-p",
        default="/Users/bigo10295/Documents/face_attr_data/26-01-19/exp_off",
        help="数据根目录，结构为 id/label(1_2,2_2)/img",
    )
    parser.add_argument(
        "--tile",
        nargs=2,
        type=int,
        default=[TARGET_TILE[0], TARGET_TILE[1]],
        metavar=("W", "H"),
        help="每张图缩放尺寸，默认 540 960",
    )
    parser.add_argument("--save", default="", help="保存整张可视化大图的路径（可选）")
    parser.add_argument("--no-show", action="store_true", help="不弹窗显示")
    args = parser.parse_args()

    root = args.path
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")

    tile_size = (int(args.tile[0]), int(args.tile[1]))  # (W, H)
    rows = build_rows(root, tile_size=tile_size)

    # 每 5 个 id 组成一页 6x10 网格（5 行 x 10 列）
    page_rows = 5
    pages = [rows[i : i + page_rows] for i in range(0, len(rows), page_rows)]
    if not pages:
        print("未找到任何图片可展示。")
        return

    base_save = args.save or ""
    for page_idx, page in enumerate(pages, start=1):
        if base_save and len(pages) > 1:
            root_name, ext = os.path.splitext(base_save)
            save_path = f"{root_name}_page{page_idx:02d}{ext or '.png'}"
        else:
            save_path = base_save or None

        render_grid(
            page,
            out_path=save_path,
            show=(not args.no_show),
            force_nrows=page_rows,
        )


if __name__ == "__main__":
    main()