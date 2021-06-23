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


def tensor2np(img):
    if isinstance(img, torch.Tensor):
        img = img.squeeze().detach().cpu().numpy()
        img = img.transpose(1, 2, 0) if len(img.shape) == 3 else img
    elif isinstance(img, np.ndarray):
        img = img.squeeze()
    else:
        raise (RuntimeError("only handle torch.Tensor or np.ndarray.\n"))
    return img


def plt_show(img):
    img = tensor2np(img)
    assert len(img.shape) == 2 or len(img.shape) == 3
    print(img.shape)
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
        # plt.imsave(path, img, cmap='gray')
    plt.show()


def cv2_show(img):
    img = tensor2np(img)
    assert len(img.shape) == 2 or len(img.shape) == 3
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def mask2img(img, mask):
    img, mask = tensor2np(img), tensor2np(mask)
    assert len(img.shape) == 2 or len(img.shape) == 3
    assert len(mask.shape) == 2
    if len(img.shape) == 2:
        img[:, :][mask[:, :] > 0] = 100
    else:
        img[:, :, 0][mask[:, :] > 0] = 255
    return img


def hist_show(img):
    img = tensor2np(img)
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.hist(img.ravel(), 256, [0, 256], color='r')
    plt.show()


def plot_loss(epoch, train_loss, val_loss, fig_name):
    fig = plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(epoch), train_loss, 'ob--', label='train loss')
    plt.plot(range(epoch), val_loss, 'or-', label='val loss')
    plt.legend()
    # plt.show()
    plt.savefig(fig_name)


if __name__ == '__main__':
    img = cv2.imread('../images/img.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('../images/mask.png', cv2.IMREAD_GRAYSCALE)
    # img_mask = mask2img(img, mask)
    hist_show(img)
    plt_show(img)
    cv2_show(mask)

    # =============== plot loss ==================
    # import numpy as np
    # train_loss = np.random.random(20) * 10
    # val_loss = np.random.random(20) * 10
    # fig_name = '../loss.png'
    # plot_loss(20, train_loss, val_loss, fig_name)
