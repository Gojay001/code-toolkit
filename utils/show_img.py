#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2021/5/18 上午12:11
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : show_img.py

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch


def plt_show(img=np.random.randn(224, 224, 3)):
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()


def cv2_show(img=np.random.randn(224, 224, 3)):
    assert len(img.shape) == 2 or len(img.shape) == 3
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def mask2img(img, mask):
    assert len(img.shape) == 2 or len(img.shape) == 3
    assert len(mask.shape) == 2
    if len(img.shape) == 2:
        img[:, :][mask[:, :] > 0] = 100
    else:
        img[:, :, 0][mask[:, :] > 0] = 255
    return img


def hist_show(img):
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.hist(img.ravel(), 256, [0, 256], color='r')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../images/img.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('../images/mask.png', cv2.IMREAD_GRAYSCALE)
    # img_mask = mask2img(img, mask)
    hist_show(img)
    plt_show(img)
    cv2_show(mask)
