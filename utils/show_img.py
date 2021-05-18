import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch


def plt_show(img=np.random.randn(224, 224, 3)):
    if len(img.shape) == 3:
        plt.imshow(img)
    elif len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    plt.show()


def cv2_show(img=np.random.randn(224, 224, 3)):
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('../images/img.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('../images/mask.png', cv2.IMREAD_GRAYSCALE)
    plt_show(img)
    cv2_show(mask)
