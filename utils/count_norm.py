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
    """MNIST: Gray, CIFAR10: RGB."""
    if data == 'mnist':
        dataset = datasets.MNIST("../data", download=True,
                        train=True, transform=transforms.ToTensor())
    elif data == 'cifar10':
        dataset = datasets.CIFAR10("../data", download=True,
                        train=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
    N_CHANNELS = 3
    mean, std = torch.zeros(1), torch.zeros(1)

    for inputs, _labels in tqdm(dataloader):
        for i in range(N_CHANNELS):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean, std)  # tensor([0.1307]) tensor([0.3015])


def voc():
    """RGB dataset"""
    means, stds = [], []
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
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    # BGR --> RGB
    means.reverse()  # [0.452, 0.431, 0.399]
    stds.reverse()  # [0.272, 0.269, 0.281]
    print(means, stds)


def imagenet():
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    print(means, stds)


def coco():
    means = [0.471, 0.448, 0.408]
    stds = [0.234, 0.239, 0.242]
    print(means, stds)

if __name__ == '__main__':
    voc()