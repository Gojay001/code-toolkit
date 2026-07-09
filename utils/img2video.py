#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author      : Gao Jie
@date        : 2024-11-19
@file        : img2video.py
@description : convert images to video and extract images from video.
@version     : 1.0
"""

import os
from tqdm import tqdm

import cv2


def imgs2video(img_path, save_path, fps=30):
    """
    convert images files to video.
    """
    img_files = [os.path.join(img_path, img) for img in os.listdir(img_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    img_files.sort()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(img_files[0])
    frame_size = (frame.shape[1], frame.shape[0])

    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

    for img in tqdm(img_files):
        img = cv2.imread(img)
        img = cv2.resize(img, frame_size)
        out.write(img)

    out.release()


def video2imgs(video_path, save_path, ori_path=None):
    """
    convert videos to image files.
    """
    if (ori_path is None) or (not os.path.exists(ori_path)):
        img_names = [f'{i:05d}.png' for i in range(1, 10000)]
    else:
        img_names = os.listdir(ori_path)
        img_names.sort()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = frame
            img_path = os.path.join(save_path, img_names[frame_id])
            cv2.imwrite(img_path, img)
            frame_id += 1

            pbar.update(1)

    cap.release()

#===========================================================

def check_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'frame_count: {frame_count}, fps: {fps}')
    cap.release()


def extract_frame(video_path, frame_id):
    """
    Extract the specified frame from a video and save it as an image.
    The output image is saved next to the video file:
        <video_name>_frame_<frame_id:05d>.png
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_id < 0 or frame_id >= total_frames:
        print(f"Error: frame_id out of range. frame_id={frame_id}, total_frames={total_frames}")
        cap.release()
        return None

    # Jump to target frame and read it.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"Error: Failed to read frame {frame_id} from {video_path}")
        return None

    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(video_dir, f"{video_name}_frame_{frame_id:05d}.png")
    cv2.imwrite(save_path, frame)
    print(f"Saved frame {frame_id} to: {save_path}")


#===========================================================

if __name__ == '__main__':
    # imgs2video('../data/celeba/train', '../video/celeba.mp4')
    # imgs2video('/Users/bigo10295/Downloads/seq/you-huoyan', '/Users/bigo10295/Downloads/seq/you-huoyan.mp4', fps=24)

    # video2imgs('../video/celeba_res.mp4', '../data/celeba/gt', '../data/celeba/train')
    # video2imgs('/Users/bigo10295/Downloads/seq/you-huoyan-red.mov', '/Users/bigo10295/Downloads/seq/you-huoyan-red')

    # check_frame('../video/celeba_res.mp4')

    extract_frame('/Users/bigo10295/Documents/3d-facemesh-test-videos/zhangfeiqian_540x960.mp4', 82)
