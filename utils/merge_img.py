import os
from tqdm import tqdm

import cv2


if __name__ == '__main__':
    base_dir = '/Users/bigo10295/Downloads/temp_img/swap_cmp/'
    folder_name_list = [base_dir + '0.' + str(i) for i in range(2,9)]

    for folder_name in tqdm(folder_name_list):
        img_dict = {}
        concat_img_list = []
        for img_name in os.listdir(folder_name):
            if img_name.split('_')[0] not in img_dict:
                img_dict[img_name.split('_')[0]] = img_name
            else:
                img1 = cv2.imread(os.path.join(folder_name, img_name))
                img2 = cv2.imread(os.path.join(folder_name, img_dict[img_name.split('_')[0]]))
                res_img = cv2.hconcat([img1, img2]) if '_dy' not in img_name else cv2.hconcat([img2, img1])
                concat_img_list.append(res_img)

        # concat concat_img_list vertically
        res_img = cv2.vconcat(concat_img_list)
        dst_path = os.path.join(folder_name, 'concat.png')
        cv2.imwrite(dst_path, res_img)
