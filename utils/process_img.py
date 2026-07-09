import os
from tqdm import tqdm
import multiprocessing

import numpy as np
import cv2


def sharpen_bilateral_usm(bgr: np.ndarray, sigma=2, coeff=2.0, sigma_color=10, sigma_space=1):
  # usm + bilateralFilter
  gaussian = cv2.GaussianBlur(bgr, (3, 3), sigma)
  bfilter = cv2.bilateralFilter(src=bgr, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)
  usm = bfilter + coeff * (bgr.astype(np.float32) - gaussian.astype(np.float32))
  return usm.clip(0, 255).astype(np.uint8)

def sharpen_eyelid_merged(args):
    src_path, dst_path = args

    img = cv2.imread(src_path)
    left_eye = img[:64, :, :]
    right_eye = img[64:, :, :]

    left_eye = sharpen_bilateral_usm(left_eye)
    right_eye = sharpen_bilateral_usm(right_eye)
    res_img = np.concatenate([left_eye, right_eye], axis=0)

    cv2.imwrite(dst_path, res_img)

def sharpen_eyelid_dataset():
    data_list = [
        # '/cephFS/gaojie/eyelid_gan/face/celeba/org/train/sharpen_double_eyelid_aligned_merged',
        '/cephFS/gaojie/eyelid_gan/face/eceleb/org/train/sharpen_double_eyelid_aligned_merged',
        # '/cephFS/gaojie/eyelid_gan/face/ffhq/org/train/sharpen_double_eyelid_aligned_merged',
        '/cephFS/gaojie/eyelid_gan/face/model/org/train/sharpen_double_eyelid_aligned_merged',
    ]

    for data_path in data_list:
        print(f'Processing {data_path}')

        src_dir = data_path
        dst_dir = data_path + '_sharpen'
        os.makedirs(dst_dir, exist_ok=True)

        img_names = os.listdir(src_dir)
        args = [(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name)) for img_name in img_names]

        with multiprocessing.Pool(processes=16) as pool:
            _ = list(tqdm(pool.imap(sharpen_eyelid_merged, args), total=len(img_names)))

# ----------------------------------------------------------

def generate_mask(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    print(img_path, img.shape)

    if img.shape[2] == 4:
        mask = img[:, :, 3]
    elif img.shape[2] == 3:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mask = img

    save_path = img_path.replace('imgs/', 'mask/')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, mask)

# ----------------------------------------------------------

def resize_image(image: np.ndarray, target_size=(256, 256)):
    """
    Resize the image to the target size.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def resize_images_mp(args):
    src_path, dst_path, dst_size = args
    img = cv2.imread(src_path)
    resized_img = resize_image(img, dst_size)
    cv2.imwrite(dst_path, resized_img)

def resize_image_dataset():
    data_list = [
        '/cephFS/gaojie/face_swap/eceleb_to_model_sr',
        '/cephFS/gaojie/face_swap/model_to_eceleb_sr',
    ]

    for data_path in data_list:
        print(f'Processing {data_path}')

        src_dir = data_path
        dst_dir = data_path + '_256'
        os.makedirs(dst_dir, exist_ok=True)

        img_names = os.listdir(src_dir)
        args = [(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name), (256, 256)) for img_name in img_names]

        with multiprocessing.Pool(processes=16) as pool:
            _ = list(tqdm(pool.imap(resize_images_mp, args), total=len(img_names)))


if __name__ == '__main__':

    # sharpen_eyelid_dataset()

    # img = cv2.imread('/cephFS/gaojie/beard_gan/stylegan/dy/val/non-beard-aligned/49.png')
    # img_sharpen = sharpen_bilateral_usm(img)
    # cv2.imwrite('result.png', img_sharpen)

    # resize_image_dataset()

    # base_dir = '/Users/bigo10295/Downloads/test_data/0814/cmp'
    # img_names = os.listdir(base_dir)
    # for img_name in img_names:
    #     if '_sr0' not in img_name:
    #         img = cv2.imread(os.path.join(base_dir, img_name))
    #         usm_img = sharpen_bilateral_usm(img)

    #         cv2.imwrite(os.path.join(base_dir, img_name.replace('.png', '_sharpen.png')), usm_img)

    # img_dir = '/Users/bigo10295/Downloads/test_data/imgs'
    # img_names = os.listdir(img_dir)
    # for img_name in img_names:
    #     generate_mask(os.path.join(img_dir, img_name))

    img_path = '../imgs/mask_face.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(img_path.replace('.png', '_256.png'), img)
