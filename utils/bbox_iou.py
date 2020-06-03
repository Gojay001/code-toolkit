import torch
import numpy as np

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    iou: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy() # If input is ndarray, turn the iou back to ndarray when return
    else:
        out_fn = lambda x: x # return torch.DoubleTensor

    # boxes_areas = boxes_width * boxes_height
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
            (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
            (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    # overlap_areas = (min_x2 - max_x1) * (min_y2 - max_y1)
    # union_areas = box1_areas + box2_areas - overlap_areas
    overlap_width = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - \
        torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    overlap_height = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - \
        torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    overlap_areas = overlap_width * overlap_height
    union_areas = box_areas.view(-1, 1) + query_areas.view(1, -1) - overlap_areas

    # print(boxes[:, 2], boxes[:, 2:3]) # [size 2], [size 2x1]
    iou = overlap_areas / union_areas
    return out_fn(iou)

if __name__ == '__main__':
    list1 = [[269.0, 195.0, 361.0, 417.0], [122.0, 125.0, 175.0, 225.0]]
    list2 = [[266.0, 196.0, 355.0, 414.0], [125.0, 126.0, 170.0, 200.0]]
    print("list1:", type(list1))
    person = np.asarray(list1).reshape(-1, 4) # <class 'numpy.ndarray'>
    head = np.asarray(list2).reshape(-1, 4)    
    print("person:", type(person), person.shape)
    iou = bbox_overlaps(person, head)
    print("IoU:", iou)