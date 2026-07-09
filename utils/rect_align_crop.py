import math
import numpy as np
import cv2


def rect_align_crop(image, bbox, input_size=224, offset=(-0.1, -0.5, 0.1, 0.1),
                    left_eye=None, right_eye=None, nose_tip=None):
    """
    Apply C++-style rect-align crop and rotation based on bbox and landmarks.

    - bbox: [x1, y1, x2, y2]
    - offset: (ox1, oy1, ox2, oy2) applied relative to width/height
    - landmarks: left_eye, right_eye, nose_tip as (x, y) in image coords
    Returns cropped and aligned image of size (input_size, input_size, 3)
    """

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Apply offset
    x1 = x1 + offset[0] * w
    y1 = y1 + offset[1] * h
    x2 = x2 + offset[2] * w
    y2 = y2 + offset[3] * h

    # Square box around center
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_side = max(x2 - x1, y2 - y1) / 2.0
    x1 = int(round(cx - half_side))
    x2 = int(round(cx + half_side))
    y1 = int(round(cy - half_side))
    y2 = int(round(cy + half_side))

    # Clamp to image bounds
    ih, iw = image.shape[:2]
    x1 = max(0, min(iw - 1, x1))
    x2 = max(0, min(iw, x2))
    y1 = max(0, min(ih - 1, y1))
    y2 = max(0, min(ih, y2))

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Compute rotation based on eyes; fall back to 0 if missing
    angle = 0.0
    if left_eye is not None and right_eye is not None:
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))

    # Rotation center is nose tip relative to ROI; fall back to ROI center
    if nose_tip is not None:
        center = (nose_tip[0] - x1, nose_tip[1] - y1)
    else:
        center = ((x2 - x1) / 2.0, (y2 - y1) / 2.0)

    size = int(max(roi.shape[0], roi.shape[1]))
    scale = float(input_size) / float(size if size > 0 else input_size)

    # Build OpenCV affine transform
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    # Shift to place center at image center
    tx = int(center[0]) - input_size // 2
    ty = int(center[1]) - input_size // 2
    rot_mat[0, 2] -= tx
    rot_mat[1, 2] -= ty

    aligned = cv2.warpAffine(roi, rot_mat, (input_size, input_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return aligned
