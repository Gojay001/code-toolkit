from __future__ import absolute_import

import numpy as np
import torch


def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # indexes by decending sort of scores

    keep = [] # result indexes
    while order.size > 0:
        # 1. Keep the b-box with largest score
        i = order.item(0)
        keep.append(i)

        # Compute the overlap ordinates between box[i] and others
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute the overlap areas
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # 2. Compute the IoU with each b-box
        iou = intersection / (areas[i] + areas[order[1:]] - intersection) # [N-1,]

        # 3. keep remain b-boxes to order
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return torch.IntTensor(keep)


if __name__ == '__main__':
    bbox_list = [[187, 82, 337, 317, 0.9], [150, 67, 305, 282, 0.75], [246, 121, 368, 304, 0.8]]
    bbox = torch.tensor(bbox_list)
    keep = nms_cpu(bbox, thresh=0.5).numpy()
    for i in range(0, keep.size):
        print(bbox_list[i])
