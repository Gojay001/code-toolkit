#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2022/4/25 19:58
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : count_norm.py

import os
from tqdm import tqdm
from time import time
import cv2
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def dataset(data='mnist', dataset=None):
    """MNIST: Gray datasets; CIFAR10: RGB datasets."""
    if data == 'mnist':
        dataset = datasets.MNIST("../data", download=True,
                        train=True, transform=transforms.ToTensor())
    elif data == 'cifar10':
        dataset = datasets.CIFAR10("../data", download=True,
                        train=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
    N_CHANNELS = 1 if data == 'mnist' else 3
    mean, std = torch.zeros(1), torch.zeros(1)

    for inputs, _labels in tqdm(dataloader):
        for i in range(N_CHANNELS):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean = mean.div_(len(dataset)).numpy().tolist()
    std = std.div_(len(dataset)).numpy().tolist()
    print(mean, std)  # [0.131] [0.302]


def voc():
    """RGB datasets."""
    mean, std = [], []
    img_list = []
    img_h, img_w = 37, 50
    N_CHANNELS = 3
    imgs_path = 'E:/data/VOCdevkit/VOC2012/JPEGImages'
    imgs_path_list = os.listdir(imgs_path)

    for item in tqdm(imgs_path_list):
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, None]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)  # [H,W,C,N]
    imgs = imgs.astype(np.float32) / 255.

    for i in range(N_CHANNELS):
        pixels = imgs[:, :, i, :].ravel()  # [WHN]
        mean.append(np.mean(pixels))
        std.append(np.std(pixels))

    # BGR --> RGB
    mean.reverse()  # [0.452, 0.431, 0.399]
    std.reverse()  # [0.272, 0.269, 0.281]
    print(mean, std)


def imagenet():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print(mean, std)


def coco():
    mean = [0.471, 0.448, 0.408]
    std = [0.234, 0.239, 0.242]
    print(mean, std)


if __name__ == '__main__':
    dataset('mnist')