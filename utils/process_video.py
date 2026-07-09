import os
import cv2
import math
import numpy as np
from tqdm import tqdm


def cut_single_video(input_path, output_path, frame_count):
    cmd = f'ffmpeg -i {input_path} -vf "select=lt(n\,{frame_count}),setpts=N/30/TB" -r 30 {output_path}'
    os.system(cmd)

def cut_video(input_path, output_path, frame_split, loop=1):
    for i in range(loop):
        if i < loop - 1:
            cmd = f'ffmpeg -i {input_path} -vf "select=between(n\,{i*frame_split}\,{(i+1)*frame_split-1}),setpts=N/30/TB" -r 30 {output_path[i]}'
        else:
            cmd = f'ffmpeg -i {input_path} -vf "select=between(n\,{i*frame_split}\,{(i+1)*frame_split}),setpts=N/30/TB" -r 30 {output_path[i]}'
        os.system(cmd)

def change_fps(input_path, output_path, fps):
    cmd = f'ffmpeg -i {input_path} -r {fps} {output_path}'
    os.system(cmd)

def generate_black_frames(output_path, size=1024, time=2):
    cmd = f'ffmpeg -f lavfi -i color=c=black:s={size}x{size}:r=30 -t {time} {output_path}'
    os.system(cmd)

def fill_black_frame(black_path, input_path, output_path):
    # cmd1 = f'ffmpeg -f lavfi -i color=c=black:s=1024x1024:r=30 -t 3 data_video/black.mp4'
    cmd2 = f'ffmpeg -i {black_path} -i {input_path} -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0" {output_path}'
    # os.system(cmd1)
    os.system(cmd2)

def convert_h264(input_path, output_path):
    cmd = f'ffmpeg -i {input_path} -c:v libx264 {output_path}'
    os.system(cmd)

def resize_720p(frame):
    frame = cv2.resize(frame, (720, 720))
    black_image = np.zeros((1280, 720, 3), dtype=np.uint8)
    start_y = (1280 - 720) // 2
    black_image[start_y:start_y + 720, :] = frame

    return black_image

def resize_and_repeat(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (720, 1280))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    black_frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = resize_720p(frame)
            for i in range(40):
                out.write(img)
            for i in range(20):
                out.write(black_frame)

            pbar.update(1)


def decrease_fps(input_path, output_path, fps):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (720, 1280))

    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0

    with tqdm(total=frame_num) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 2 == 0:
                out.write(frame)
                # for _ in range(2):
                #     out.write(frame)

            frame_count += 1
            pbar.update(1)

    out.release()

#=================================================================================
def cut_and_merge():
    # cut_single_video('data_video/celeba/celeba_repeat_non_eyebrow_1.mp4', 'data_video/celeba/celeba_non_eyebrow_1.mp4', 5940)
    # cut_single_video('data_video/celeba/celeba_repeat_non_eyebrow_6.mp4', 'data_video/celeba/celeba_non_eyebrow_6.mp4', 1170)

    cap = cv2.VideoCapture('/Users/bigo10295/Documents/image_pair_selector/MegaFace.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'total frame: {frame_count}, fps: {int(cap.get(cv2.CAP_PROP_FPS))}')

    loop = math.ceil(frame_count / 4200.0)
    output_path_list = [f'/Users/bigo10295/Documents/image_pair_selector/MegaFace_{i}.mp4' for i in range(loop)]
    cut_video('/Users/bigo10295/Documents/image_pair_selector/MegaFace.mp4', output_path_list, 4200, loop)

    # ------------------------------------------------------------------
    # generate_black_frames('data_video/black_2s.mp4', time=2)

    # for i in range(7):
    #     fill_black_frame('data_video/black_2s.mp4', f'data_video/celeba/celeba_{i}.mp4', f'data_video/celeba/celeba_black_{i}.mp4')

def repeat_video():
    for i in range(31):
        resize_and_repeat(f'data_video/celeba/celeba_{i}.mp4', f'data_video/celeba/celeba_repeat_{i}.mp4')


if __name__ == '__main__':
    # cut_and_merge()

    # decrease_fps('/Users/bigo10295/Downloads/optical_flow/assets/run.mp4', '/Users/bigo10295/Downloads/optical_flow/assets/run_15.mp4', 15)
    # decrease_fps('/Users/bigo10295/Documents/3d-facemesh-test-videos/chenwei_720x1280.mp4', '/Users/bigo10295/Downloads/optical_flow/assets/chenwei_720p.mp4', 15)

    # repeat_video()

    # convert_h264('/Users/bigo10295/Downloads/nggyup_22_sr.mp4', '/Users/bigo10295/Downloads/nggyup_22_sr_h264.mp4')
    # convert_h264('/Users/bigo10295/Downloads/temp_img/liuyifei_100.mp4', '/Users/bigo10295/Downloads/temp_img/result_liuyifei_100.mp4')

    # img = cv2.imread('/Users/bigo10295/Downloads/temp_img/swap_cmp/celeba/28375.jpg')
    # adjust_img = resize_720p(img)
    # cv2.imwrite('/Users/bigo10295/Downloads/temp_img/swap_cmp/adjust_img.png', adjust_img)

    change_fps('/Users/bigo10295/Downloads/temp_img/xs.mp4', '/Users/bigo10295/Downloads/temp_img/xs_10.mp4', 10)

