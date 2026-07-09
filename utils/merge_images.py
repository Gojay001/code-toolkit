import os
import cv2
import numpy as np
from tqdm import tqdm

from datetime import datetime


def merge_two_imgs(img_dirs, img_name, result_dir):
    image1 = cv2.imread(os.path.join(img_dirs[0], img_name))
    image2 = cv2.imread(os.path.join(img_dirs[1], img_name))

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    merged_image = np.hstack((image1, image2))

    cv2.imwrite(os.path.join(result_dir, img_name), merged_image)


def merge_three_imgs(img_name):
    image1 = cv2.imread(f'hot/{img_name}')
    image2 = cv2.imread(f'res_imgs_dilate/{img_name}')
    image3 = cv2.imread(f'res_imgs_dilate_s_2220/{img_name}')

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))

    merged_image = np.hstack((image1, image2, image3))

    cv2.imwrite(f'merge/{img_name}', merged_image)

def merge_four_imgs(img_name):
    image1 = cv2.imread(f'hot/{img_name}')
    image2 = cv2.imread(f'res_imgs_dy_data_24/{img_name}')
    image3 = cv2.imread(f'res_imgs_dy_blend_24/{img_name}')
    image4 = cv2.imread(f'hot_dy/{img_name}')

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))
    image4 = cv2.resize(image4, (image1.shape[1], image1.shape[0]))

    merged_image = np.hstack((image1, image2, image3, image4))

    cv2.imwrite(f'merge/{img_name}', merged_image)

def merge_five_imgs(img_name):
    # image1 = cv2.imread(f'hot/{img_name}')
    # image2 = cv2.imread(f'res_imgs_lapa_ffhq/{img_name}')
    # image3 = cv2.imread(f'res_imgs_lapa_ffhq_live/{img_name}')
    # image4 = cv2.imread(f'res_imgs_lapa_ffhq_live_308/{img_name}')
    # image5 = cv2.imread(f'hot_dy/{img_name}')
    # image5 = cv2.imread(f'res_imgs_lapa_ffhq_live_208/{img_name}')
    # image5 = cv2.imread(f'res_imgs_lapa_ffhq_live_101/{img_name}')

    image1 = cv2.imread(f'hot/{img_name}')
    # image2 = cv2.imread(f'res_imgs_blend_150/{img_name}')
    # image3 = cv2.imread(f'res_imgs_hair/{img_name}')
    # image4 = cv2.imread(f'res_imgs_dilate_s/{img_name}')
    image2 = cv2.imread(f'res_imgs_dilate_s_2420_stable/{img_name}')
    image3 = cv2.imread(f'res_imgs_dy_data_24/{img_name}')
    image4 = cv2.imread(f'res_imgs_dy_blend_24/{img_name}')
    image5 = cv2.imread(f'hot_dy/{img_name}')

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))
    image4 = cv2.resize(image4, (image1.shape[1], image1.shape[0]))
    image5 = cv2.resize(image5, (image1.shape[1], image1.shape[0]))

    merged_image = np.hstack((image1, image2, image3, image4, image5))

    cv2.imwrite(f'merge/{img_name}', merged_image)

def merge_six_imgs(img_name):
    image1 = cv2.imread(f'hot/{img_name}')
    # image2 = cv2.imread(f'res_imgs_tt/{img_name}')
    # image3 = cv2.imread(f'res_imgs_dy/{img_name}')
    # image4 = cv2.imread(f'res_imgs_venus/{img_name}')
    image2 = cv2.imread(f'res_imgs_dilate_s_2420_stable/{img_name}')
    image3 = cv2.imread(f'res_imgs_dy_data_22/{img_name}')
    image4 = cv2.imread(f'res_imgs_dy_data_24/{img_name}')
    image5 = cv2.imread(f'res_imgs_dy_22/{img_name}')
    image6 = cv2.imread(f'hot_dy/{img_name}')

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))
    image4 = cv2.resize(image4, (image1.shape[1], image1.shape[0]))
    image5 = cv2.resize(image5, (image1.shape[1], image1.shape[0]))
    image6 = cv2.resize(image6, (image1.shape[1], image1.shape[0]))

    merged_image = np.hstack((image1, image2, image3, image4, image5, image6))

    cv2.imwrite(f'merge/{img_name}', merged_image)

def merge_seven_imgs(img_name):
    image1 = cv2.imread(f'hot/{img_name}')
    image2 = cv2.imread(f'res_imgs_dilate_s_2420_stable/{img_name}')
    image3 = cv2.imread(f'res_imgs_dy_data_22/{img_name}')
    image4 = cv2.imread(f'res_imgs_dy_data_24/{img_name}')
    image5 = cv2.imread(f'res_imgs_dy_22/{img_name}')
    image6 = cv2.imread(f'res_imgs_dy_24/{img_name}')
    image7 = cv2.imread(f'hot_dy/{img_name}')

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))
    image4 = cv2.resize(image4, (image1.shape[1], image1.shape[0]))
    image5 = cv2.resize(image5, (image1.shape[1], image1.shape[0]))
    image6 = cv2.resize(image6, (image1.shape[1], image1.shape[0]))
    image7 = cv2.resize(image7, (image1.shape[1], image1.shape[0]))

    merged_image = np.hstack((image1, image2, image3, image4, image5, image6, image7))

    cv2.imwrite(f'merge/{img_name}', merged_image)
 

if __name__ == '__main__':
    # img_names_bvt = os.listdir('res_imgs_lapa_ffhq_live')
    # img_names_dlib = os.listdir('res_imgs_lapa_ffhq_live_align')
    # img_names_dy = os.listdir('hot_dy')

    # img_names_dlib = os.listdir('res_imgs_dilate_s_2420_stable')
    # img_names_bvt = os.listdir('res_imgs_dy_data_24')
    # img_names_dy = os.listdir('hot_dy')
    # img_names = list(set(img_names_bvt).intersection(img_names_dlib, img_names_dy))

    img_dirs = [
        'beard_test/beard_demo',
        'beard_test/beard_demo_result'
    ]
    
    img_names = os.listdir(img_dirs[0])

    result_dir = 'beard_test/merge'
    os.makedirs(result_dir, exist_ok=True)

    for img_name in tqdm(img_names):
    	merge_two_imgs(img_dirs, img_name, result_dir)
        # merge_three_imgs(img_name)
        # merge_four_imgs(img_name)
        # merge_five_imgs(img_name)
        # merge_six_imgs(img_name)
        # merge_seven_imgs(img_name)
