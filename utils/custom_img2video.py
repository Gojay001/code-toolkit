#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two images -> one video with a fixed alternating frame-count pattern.
"""

import argparse
import os

import cv2

# (img1 连续帧数, img2 连续帧数) 按顺序拼接
FRAME_PATTERN = [
    (5, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (4, 4),
    (3, 3),
    (2, 2),
    (1, 1),
]


def load_image_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def build_frames(img1, img2):
    """按 FRAME_PATTERN 生成帧列表（BGR ndarray）。"""
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    frames = []
    for n1, n2 in FRAME_PATTERN:
        frames.extend([img1.copy() for _ in range(n1)])
        frames.extend([img2.copy() for _ in range(n2)])
    return frames


def images_to_video(img1_path, img2_path, save_path, fps=30, fourcc="mp4v"):
    img1 = load_image_bgr(img1_path)
    img2 = load_image_bgr(img2_path)
    frames = build_frames(img1, img2)

    h, w = frames[0].shape[:2]
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(save_path, fourcc_code, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {save_path}")

    for f in frames:
        writer.write(f)
    writer.release()
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description="两张图按固定规则交替帧数合成视频")
    parser.add_argument(
        "--img1",
        default="/Users/bigo10295/Documents/3d-facemesh-test-videos/zhangfeiqian_540x960_frame_00000.png",
        help="第一张图片路径",
    )
    parser.add_argument(
        "--img2",
        default="/Users/bigo10295/Documents/3d-facemesh-test-videos/zhangfeiqian_540x960_frame_00082.png",
        help="第二张图片路径",
    )
    parser.add_argument(
        "--out",
        default="/Users/bigo10295/Documents/3d-facemesh-test-videos/zhangfeiqian_custom_pattern.mp4",
        help="输出视频路径",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="四字符编码，如 mp4v、avc1（需本机支持）",
    )
    args = parser.parse_args()

    n = images_to_video(args.img1, args.img2, args.out, fps=args.fps, fourcc=args.fourcc)
    print(f"已保存: {args.out}，总帧数: {n}，fps: {args.fps}")


if __name__ == "__main__":
    main()
