#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2021/9/15 下午5:44
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : test_net.py

import torch
from torch import nn
from torchkeras import summary
from thop import profile


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        y = self.relu(x)
        return y


if __name__ == '__main__':
    model = Net()
    print(model)
    print(summary(model, input_shape=(3, 20, 20)))
    print('number of params:', sum(param.numel() for param in model.parameters()))
    inputs = torch.randn(8, 3, 20, 20)
    flops, params = profile(model, (inputs,))
    print('flops:', flops, 'params:', params)
