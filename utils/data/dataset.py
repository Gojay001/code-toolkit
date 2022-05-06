#!/usr/bin/env python3
# encoding: utf-8
# @Time   : 2021/5/18 上午11:46
# @Author : gojay
# @Contact: gao.jay@foxmail.com
# @File   : dataset.py

from torch.utils.data import Dataset

import os
import cv2
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data_root, data_txt, transform=None):
        super(MyDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        if not os.path.isfile(data_txt):
            raise (RuntimeError("Image list file do not exist:" + data_txt + "\n"))
        with open(data_txt, 'r') as f:
            self.data_list = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_name = self.data_list[index].split()[0]
        label_name = self.data_list[index].split()[1]
        image_path = os.path.join(self.data_root, image_name)
        label_path = os.path.join(self.data_root, label_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch:" + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label


if __name__ == '__main__':
    import utils.data.seg_transform as transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pad_value = [item * 255 for item in mean]
    train_transform = transforms.Compose([
        transforms.HistEqualize(),
        transforms.RandScale([0.9, 1.1]),
        transforms.RandRotate([-10, 10], padding=pad_value, ignore_label=255),
        transforms.RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.Crop([473, 473], crop_type='rand', padding=pad_value, ignore_label=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataset = MyDataset("../data", "../data/train.txt", train_transform)
    img, label = train_dataset[0]
    print(img, img.shape)
    print(label, label.shape)
    label = label.numpy()
    print(np.sum(label == 255))
