#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2021/8/30 下午12:09
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : CrossEntropyLoss.py

import torch
import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=[0.5, 0.25, 0.25], reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward_0(self, x=torch.rand(4, 3), target=torch.tensor([2, 1, 0, 1])):
        """
        Formula implementation
        x: (N, C) or (N, C, H, W) or (N, C, d1, ..., dk)
        target: (N) or (N, H, W) or (N, d1, ..., dk)
        """
        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim=1).reshape(4, 1).repeat(1, 3)
        x_softmax = x_exp / x_exp_sum
        x_log_softmax = torch.log(x_softmax)
        print('x:', x, '\n', 'x_exp:', x_exp, '\n', 'x_exp_sum:', x_exp_sum,
              '\n', 'x_softmax:', x_softmax, '\n', 'x_log_softmax:', x_log_softmax)
        nll_loss = 0
        for i in range(len(target)):
            if self.weight:
                nll_loss += -x_log_softmax[i][target[i]] * self.weight[target[i]]
            else:
                nll_loss += -x_log_softmax[i][target[i]]
        if self.reduction == 'mean':
            return nll_loss / len(target)
        elif self.reduction == 'sum':
            return nll_loss

    def forward(self, x=torch.rand(4, 3), target=torch.tensor([2, 1, 0, 1])):
        """
        Simplified formula implementation
        x: (N, C) or (N, C, H, W) or (N, C, d1, ..., dk)
        target: (N) or (N, H, W) or (N, d1, ..., dk)
        """
        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim=1)
        x_log = torch.log(x_exp_sum)
        nll_loss = 0
        for i in range(len(target)):
            nll_loss += -x[i][target[i]] + x_log[i]
        if self.reduction == 'mean':
            return nll_loss / len(target)
        elif self.reduction == 'sum':
            return nll_loss


class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward_0(self, x, target):
        """
        Formula implementation
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        """
        x_sigmoid = torch.sigmoid(x)
        bce_loss = torch.where(target == 1, -torch.log(x_sigmoid),
                               -torch.log(torch.ones_like(x_sigmoid) - x_sigmoid))
        return bce_loss.mean()

    def forward(self, x, target):
        """
        Simplified formula implementation
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        """
        x_sigmoid = torch.where(
            target == 1, torch.sigmoid(x), torch.sigmoid(-x))
        bce_loss = -torch.log(x_sigmoid)
        return bce_loss.mean()


if __name__ == '__main__':
    # Multi-class: 4 samples, and 3 categories
    x = torch.rand((4, 3), requires_grad=True)
    y = torch.randint(0, 3, (4,))
    print(x, y)
    # Two-class
    x1 = torch.rand((1, 2, 32, 32), requires_grad=True)
    y1 = torch.randint(0, 2, (1, 2, 32, 32)).float()
    ce_cus = CustomCrossEntropyLoss(weight=None, reduction='mean')
    ce_nn = nn.CrossEntropyLoss()
    bce_cus = CustomBCELoss()
    bce_nn_logit = nn.BCEWithLogitsLoss()
    bce_nn = nn.BCELoss()
    loss1 = ce_cus(x, y)
    loss2 = ce_nn(x, y)
    loss3 = bce_cus(x1, y1)
    loss4 = bce_nn_logit(x1, y1)
    loss5 = bce_nn(x1, y1)
    print(loss1, loss2, loss3, loss4, loss5)
