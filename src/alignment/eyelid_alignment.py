import numpy as np
import scipy.ndimage
import os
import PIL.Image

def get_crop_rect_bvt(face_landmarks, expansion_ratio=1.0, crop_eye="left_eye"):
    # bvt landmarks
    lm = np.array(face_landmarks)
    lm_chin = lm[0: 33]  # left-right
    lm_eyebrow_left_up = lm[33: 38]  # left-right
    lm_eyebrow_right_up = lm[38: 43]  # left-right
    lm_nose = lm[43: 47]  # top-down
    lm_nostrils = lm[47: 52]  # left-right
    lm_eye_left = lm[52: 58]  # left-clockwise
    lm_eye_right = lm[58: 64]  # left-clockwise
    lm_eyebrow_left_down = lm[64: 68]  # left-clockwise
    lm_eyebrow_right_down = lm[68: 72]  # left-clockwise
    lm_eye_left_mid = lm[72: 75]  # left-clockwise
    lm_eye_right_mid = lm[75: 78]  # left-clockwise
    lm_nose_outer = lm[78: 84]  # left-clockwise
    lm_mouth_outer = lm[84: 97]  # left-clockwise
    lm_mouth_inner = lm[97: 104]  # left-clockwise
    lm_eyeball_left = lm[104]  # left-clockwise
    lm_eyeball_right = lm[105]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    # x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    # x /= np.hypot(*x)

    left_eye_length = np.hypot(*(lm[55] - lm[52]))
    right_eye_length = np.hypot(*(lm[61] - lm[58]))

    max_eye_length = max(left_eye_length, right_eye_length)

    if crop_eye == "left_eye":

        # Choose oriented crop rectangle.
        x = lm[55] - lm[52]
        x /= np.hypot(*x)

        x *= 0.9 * max(left_eye_length, 0.8 * max_eye_length)

        x *= expansion_ratio

        y = np.flipud(x) * [-1, 1] * 0.5

        # c = (lm[52] + lm[55]) * 0.5
        c = lm[72]

    else:

        # Choose oriented crop rectangle.
        x = lm[61] - lm[58]
        x /= np.hypot(*x)

        x *= 0.9 * max(right_eye_length, 0.8 * max_eye_length)

        x *= expansion_ratio

        y = np.flipud(x) * [-1, 1] * 0.5

        # c = (lm[58] + lm[61]) * 0.5
        c = lm[75]

    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    return quad, qsize

def image_align_run(img, face_landmarks, gt_img = None, mask_img = None,  detector='bvt', output_size=1024, transform_size=4096, enable_padding=False, expansion_ratio=1.0, crop_eye="left_eye"):

    assert detector == 'bvt', 'Unsupported detector: {}'.format(detector)

    # Align function from FFHQ dataset pre-processing step
    quad, qsize = get_crop_rect_bvt(face_landmarks, expansion_ratio, crop_eye)

    info = {}
    info['ori_quad'] = np.copy(quad)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    info['shrink'] = shrink
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))

        img = img.resize(rsize, PIL.Image.LANCZOS)
        if gt_img:
            gt_img = gt_img.resize(rsize, PIL.Image.LANCZOS)
        if mask_img:
            mask_img = mask_img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    info['crop'] = crop
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:

        img = img.crop(crop)

        if gt_img:
            gt_img = gt_img.crop(crop)
        if mask_img:
            mask_img = mask_img.crop(crop)

        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))

        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')

        if gt_img:
            gt_img = np.pad(np.float32(gt_img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        if mask_img:
            mask_img = np.pad(np.float32(mask_img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')

        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    if gt_img:
        gt_img = gt_img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if mask_img:
        mask_img = mask_img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    if output_size < transform_size:

        output_res = (output_size, output_size // 2)

        resized_output_img = img.resize(output_res, PIL.Image.LANCZOS)

        if gt_img:
            resized_gt_img = gt_img.resize(output_res, PIL.Image.LANCZOS)
        if mask_img:
            resized_mask_img = mask_img.resize(output_res, PIL.Image.LANCZOS)

    info['transform_size'] = transform_size
    info['output_size'] = output_size
    info['quad'] = (quad + 0.5).flatten()

    if gt_img and mask_img:
        return info, resized_output_img, resized_gt_img, resized_mask_img
    elif gt_img:
        return info, resized_output_img, resized_gt_img
    else:
        return info, resized_output_img