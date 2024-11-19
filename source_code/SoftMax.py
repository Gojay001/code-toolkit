#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2021/8/30 上午11:40
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : SoftMax.py

import numpy as np


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


def shift_softmax(x):
    shift = x - np.max(x)
    exps = np.exp(shift)
    return exps / np.sum(exps)


def log_softmax(x, recover_probs=True):
    shift = x - np.max(x)
    exps = np.exp(shift)
    sum_exps = np.sum(exps)
    log_sum_exps = np.log(sum_exps)
    log_probs = x - np.max(x) - log_sum_exps
    # recover probs
    if recover_probs:
        exp_log_probs = np.exp(log_probs)
        sum_log_probs = np.sum(exp_log_probs)
        probs = exp_log_probs / sum_log_probs
        return probs
    return log_probs


if __name__ == '__main__':
    x = np.array([10, 2, 40, 4])  # [9.35762297e-14 3.13913279e-17 1.00000000e+00 2.31952283e-16]
    x_out = np.array([10, 2, 10000, 4])  # [ 0.  0. nan  0.] -> 数值不稳定
    print(softmax(x))
    print(shift_softmax(x_out))
    print(log_softmax(x_out))
