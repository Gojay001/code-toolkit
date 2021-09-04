#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2021/8/31 下午3:29
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : regularization.py

import torch
import numpy as np


# L1 norm, L2 norm
def norm(x=torch.rand(3, 4)):
    y_norm1 = torch.sum(torch.abs(x), dim=(1, ), keepdim=True)
    y_norm2 = torch.sqrt(torch.sum(torch.pow(x, 2), dim=(1, ), keepdim=True))
    y_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    print(x, y_norm1, y_norm2, y_norm)


# L1 regularization
def l1_reg(model, alpha):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss += alpha * torch.sum(torch.abs(param))
    return l1_loss


# L2 regularization
def l2_reg(model, beta):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_loss += (0.5 * beta * torch.sum(torch.pow(param, 2)))
    return l2_loss


# Dropout
def Dropout(X, p=0.5):
    W1, W2, W3 = 1, 2, 0.5
    b1, b2, b3 = 1, 0.5, 2

    def train_step(X):
        X1 = np.maximum(0, np.dot(W1, X) + b1)
        mask1 = (np.random.rand(*X1.shape) < p) / p
        X1 *= mask1
        X2 = np.maximum(0, np.dot(W2, X1) + b2)
        mask2 = (np.random.rand(*X2.shape) < p) / p
        X2 *= mask2
        out = np.dot(W3, X2) + b3

    def predict(X):
        X1 = np.maximum(0, np.dot(W1, X) + b1)
        X2 = np.maximum(0, np.dot(W2, X1) + b2)
        out = np.dot(W3, X2) + b3


if __name__ == '__main__':
    norm()
