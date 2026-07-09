#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author      : Gao Jie
@date        : 2024-11-20
@file        : process_mask.py
@description : process mask functions
@version     : 1.0
"""

import os
import random
import multiprocessing
from tqdm import tqdm

import numpy as np
import cv2


seed = 42
random.seed(seed)

def add_half_mask_to_img(args):
    img_path, mask_path, gt_path, dst_img_path, dst_mask_path, dst_gt_path = args

    ori_img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    gt_img = cv2.imread(gt_path)

    # select random half_mask
    h, w, _ = ori_img.shape
    top, bottom = random.uniform(0.3, 0.7) * w, random.uniform(0.3, 0.7) * w

    points_left  = np.array([[(0, 0)], [(top, 0)], [(bottom, h-1)], [(0, h-1)]], dtype=np.int32)
    points_right = np.array([[(top, 0)], [(bottom, h-1)], [(w-1, h-1)], [(w-1, 0)]], dtype=np.int32)
    points = random.choice([points_left, points_right])

    half_mask = np.zeros([h, w], dtype=np.uint8)
    cv2.fillPoly(half_mask, [points], 255)

    # blend mask
    masked_img = cv2.bitwise_and(ori_img, ori_img, mask=half_mask)
    masked_mask = cv2.bitwise_and(mask, mask, mask=half_mask)
    masked_gt_img = cv2.bitwise_and(gt_img, gt_img, mask=half_mask)

    cv2.imwrite(dst_img_path, masked_img)
    cv2.imwrite(dst_mask_path, masked_mask)
    cv2.imwrite(dst_gt_path, masked_gt_img)

def add_half_mask(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst_dir = f'{dst_dir}/train'
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(f'{dst_dir}/eyebrow', exist_ok=True)
    os.makedirs(f'{dst_dir}/mask_org', exist_ok=True)
    os.makedirs(f'{dst_dir}/non-eyebrow', exist_ok=True)

    img_names = os.listdir(f'{src_dir}/eyebrow')

    args = [(f'{src_dir}/eyebrow/{img_name}', f'{src_dir}/mask_org/{img_name}', f'{src_dir}/non-eyebrow/{img_name}',
        f'{dst_dir}/eyebrow/{img_name}', f'{dst_dir}/mask_org/{img_name}', f'{dst_dir}/non-eyebrow/{img_name}') for img_name in img_names]

    with multiprocessing.Pool(processes=16) as pool:
        _ = list(tqdm(pool.imap(add_half_mask_to_img, args), total=len(img_names)))

#================================================================================

def dilate_mask(mask, dilate_size=20, blur_size=20):
    mask_dilate = cv2.dilate(mask, kernel=np.ones((dilate_size, dilate_size), np.uint8))

    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(blur_size, blur_size)) if blur_size > 0 else mask_dilate

    mask_image = (mask + (255 - mask) / 255 * mask_dilate_blur).astype(np.uint8)

    return mask_image

def dilate_mask_img(args):
    mask_path, dst_path, dilate_size, blur_size = args

    mask = cv2.imread(mask_path)
    mask_image = dilate_mask(mask, dilate_size, blur_size)

    cv2.imwrite(dst_path, mask_image)

def dilate_mask_dataset():
    src_list = [
        '../data/celeba',
        '../data/ffhq',
    ]

    for src_dir in src_list:
        dst_dir = src_dir.replace('mask', 'mask_dilate')
        os.makedirs(dst_dir, exist_ok=True)

        img_names = os.listdir(src_dir)
        args = [(f'{src_dir}/{img_name}', f'{dst_dir}/{img_name}', 20, 20) for img_name in img_names]

        with multiprocessing.Pool(processes=16) as pool:
            _ = list(tqdm(pool.imap(dilate_mask_img, args), total=len(img_names)))


if __name__ == '__main__':
    # add_half_mask('../data/celeba', '../data/celeba_occlusion')

    dilate_mask_dataset()
