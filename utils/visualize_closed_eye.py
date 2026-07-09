import sys
sys.path.append('..')

import os
import zipfile
import onnx2torch
import numpy as np
from tqdm import tqdm
import shutil
import concurrent.futures

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont


def run_attribute_networker(attribute_networker, img):
    img_norm   = (img + 1.0) / 2.0
    img_resize = F.interpolate(img_norm, size=(224,224), mode='bicubic')
    latent_attribute = attribute_networker(img_resize)
    return latent_attribute

def get_eye_bs(attribute_networker, img_path):
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    img = Image.open(img_path).convert('RGB')
    img = transform(image=np.array(img))['image']
    img = img.unsqueeze(0).cuda()

    latent_attribute = run_attribute_networker(attribute_networker, img)

    eye_blink = latent_attribute[:, 79+6:79+8]
    eye_wide  = latent_attribute[:, 79+18:79+20]

    return eye_blink[0].detach().cpu().numpy(), eye_wide[0].detach().cpu().numpy()

def draw_info_on_triple_image(source_img_path, target_img_path, gt_img_path, info_args):
    source_img = Image.open(source_img_path).convert('RGB')
    target_img = Image.open(target_img_path).convert('RGB')
    gt_img     = Image.open(gt_img_path).convert('RGB')

    #concatenate source_img and target_img and gt_img
    concat_img = Image.new('RGB', (source_img.width + target_img.width + gt_img.width, source_img.height))
    concat_img.paste(source_img, (0, 0))
    concat_img.paste(target_img, (source_img.width, 0))
    concat_img.paste(gt_img, (source_img.width + target_img.width, 0))

    #draw cls_score on concat_img
    draw = ImageDraw.Draw(concat_img)
    draw.text((0, 0), info_args[0], fill=(255, 0, 0), font_size=20)
    draw.text((0, 20), info_args[1], fill=(0, 255, 0), font_size=20)
    draw.text((0, 40), info_args[2], fill=(0, 0, 255), font_size=20)

    return concat_img

def detect_eye_closed(args):
    attribute_networker, dataset_dir, mode, gt_dataset_name, gt_img_name = args

    source_dataset_name = gt_dataset_name.split('_to_')[0].split('_')[0]
    target_dataset_name = gt_dataset_name.split('_to_')[1].split('_')[0]
    suffix = gt_img_name.split('.')[-1]
    source_img_name = gt_img_name.split('_to_')[0] + '.' + suffix
    target_img_name = gt_img_name.split('_to_')[1]
    source_img_path = os.path.join(dataset_dir, mode, source_dataset_name, source_img_name.replace('.jpg', '.png'))
    target_img_path = os.path.join(dataset_dir, mode, target_dataset_name, target_img_name.replace('.jpg', '.png'))
    gt_img_path = os.path.join(dataset_dir, mode, gt_dataset_name, gt_img_name)

    if not os.path.exists(source_img_path) or not os.path.exists(target_img_path) or not os.path.exists(gt_img_path):
        os.makedirs(os.path.join(dataset_dir, mode, gt_dataset_name + '_name'), exist_ok=True)
        shutil.move(gt_img_path, os.path.join(dataset_dir, mode, gt_dataset_name + '_name', gt_img_name))
        return

    remove_dir = os.path.join(dataset_dir, mode, gt_dataset_name + '_closed_eye')
    os.makedirs(remove_dir, exist_ok=True)
    compare_dir = os.path.join(dataset_dir, mode, gt_dataset_name + '_closed_eye_compare')
    os.makedirs(compare_dir, exist_ok=True)

    source_eye_blink, source_eye_wide = get_eye_bs(attribute_networker, source_img_path)
    target_eye_blink, target_eye_wide = get_eye_bs(attribute_networker, target_img_path)
    gt_eye_blink, gt_eye_wide = get_eye_bs(attribute_networker, gt_img_path)

    if abs(gt_eye_blink[0] - target_eye_blink[0]) > 0.2 or abs(gt_eye_blink[0] - target_eye_blink[0]) > 0.2 or \
       abs(gt_eye_wide[0] - target_eye_wide[0]) > 0.2 or abs(gt_eye_wide[0] - target_eye_wide[0]) > 0.2:
    # if True:
    # if abs(gt_eye_blink[0] - target_eye_blink[0]) > 0.3 or abs(gt_eye_blink[0] - target_eye_blink[0]) > 0.3:
        print(f'{gt_img_path} is closed eye')
        source_info = f'source eye_blink: [{source_eye_blink[0]:.2f}, {source_eye_blink[1]:.2f}], eye_wide: [{source_eye_wide[0]:.2f}, {source_eye_wide[1]:.2f}]'
        target_info = f'target eye_blink: [{target_eye_blink[0]:.2f}, {target_eye_blink[1]:.2f}], eye_wide: [{target_eye_wide[0]:.2f}, {target_eye_wide[1]:.2f}]'
        gt_info = f'gt eye_blink: [{gt_eye_blink[0]:.2f}, {gt_eye_blink[1]:.2f}], eye_wide: [{gt_eye_wide[0]:.2f}, {gt_eye_wide[1]:.2f}]'

        concat_img = draw_info_on_triple_image(source_img_path, target_img_path, gt_img_path, [source_info, target_info, gt_info])
        concat_img.save(os.path.join(compare_dir, os.path.basename(gt_img_path)))
        shutil.move(gt_img_path, os.path.join(remove_dir, os.path.basename(gt_img_path)))


def main(attribute_networker):
    dataset_dir = '/data1/gaojie/data/face_swap/deeplive'
    # dataset_dir = '/data1/gaojie/data/face_swap/douyin'
    mode = 'train'
    # gt_dataset_name = 'random_to_random'
    gt_dataset_name = 'random_more_to_random_more'
    # gt_dataset_name = 'ffhq_to_ced'
    # gt_dataset_name = 'random_weighted_to_random_weighted'

    gt_dir = os.path.join(dataset_dir, mode, gt_dataset_name)
    gt_img_names = os.listdir(gt_dir)
    args = [(attribute_networker, dataset_dir, mode, gt_dataset_name, gt_img_name) for gt_img_name in gt_img_names]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        _ = list(tqdm(executor.map(detect_eye_closed, args), total=len(args)))


if __name__ == "__main__":
    model_path = '../arcface_model/facemesh.bigo_bilinear_recon.mobilenet_v2_020.pt'

    attribute_networker = torch.load(model_path, weights_only=False)
    attribute_networker = attribute_networker.cuda()
    attribute_networker.eval()

    main(attribute_networker)

    # img_path = '/data1/gaojie/data/face_swap/deeplive/val/target/7_0015.png'
    # eye_blink, eye_wide = get_eye_bs(attribute_networker, img_path)
    # print(f'Eye Blink: {eye_blink}, Eye Wide: {eye_wide}')

    # detect_eye_closed(attribute_networker, '/data1/gaojie/data/face_swap/deeplive', 'val', 'adults_17_to_eceleb_9316.png')
