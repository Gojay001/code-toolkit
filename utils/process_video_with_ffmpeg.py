"""
    ffmpeg ops
"""
# # pad video to make it square
# ffmpeg -i trajectory_erase.mp4 -vf "pad=1440:1440:240:240:color=black" trajectory_pad.mp4

# # crop video to make it square
# ffmpeg -i input_720p.mp4 -vf "crop=ih:ih:(iw-ih)/2:0" output_720x720.mp4

# # make 1:1 video to 1s, 720x720, (change PTS and scale)
# ffmpeg -i video.mp4 -filter_complex "[0:v]setpts=0.3*PTS,scale=480:480[v]" -map "[v]" -c:v libx264 -r 24 trajectory.mp4
# ffmpeg -i fire_2_yj_3s.mp4 -vf "crop=1080:1080:0:420" fire_2_yj_3s_1080.mp4

# # remove black background from video
# ffmpeg -i trajectory.mp4 -vf "lumakey=threshold=0.1:tolerance=0.1:softness=0.5" -c:v png -pix_fmt rgba trajectory.mov

# # remove white background from video
# ffmpeg -i purple_snow.mov -vf "colorkey=white:0.1:0.05" -c:v png -pix_fmt rgba purple_snow_wo_bg.mov

# # make video.webp from video.mov
# ffmpeg -i output_trajectory.mov -framerate 24 -loop 1 -compression_level 6 -pix_fmt yuva420p output_trajectory.webp

# # extract rgba frames from video
# ffmpeg -i purple_snow_wo_bg.mov -vf "scale=360:640:flags=lanczos,format=rgba" -c:v png -pix_fmt rgba  -vsync 0 -compression_level 9 frames/frame_%03d.png
# ffmpeg -i input_video.mp4 frames/frame_%04d.png

# # 压缩webp视频到更小（方法1：降低分辨率，推荐）
# ffmpeg -i output_trajectory.mov -vf "scale=iw*0.7:ih*0.7" -framerate 24 -loop 1 -compression_level 6 -pix_fmt yuva420p output_trajectory.webp

# # 压缩webp视频到更小（方法2：降低分辨率+降低帧率）
# ffmpeg -i output_trajectory.mov -vf "scale=iw*0.6:ih*0.6" -framerate 20 -loop 1 -compression_level 6 -pix_fmt yuva420p output_trajectory.webp

# # 压缩webp视频到更小（方法3：固定分辨率，如512x512）
# ffmpeg -i output_trajectory.mov -vf "scale=512:512" -framerate 24 -loop 1 -compression_level 6 -pix_fmt yuva420p output_trajectory.webp

# ffmpeg -i fire_5_yj_jy.mov -filter_complex "[0:v]setpts=1.0*PTS,scale=480:480[v]" -map "[v]" -c:v libx264 -r 24 fire_5_yj_jy_480.mov
# ffmpeg -i fire_5_yj_jy_480.mov -vf "lumakey=threshold=0.1:tolerance=0.1:softness=0.1" -c:v png -pix_fmt rgba fire_5_yj_jy_rgba.mov
# ffmpeg -i fire_5_yj_jy_rgba.mov -framerate 24 -loop 1 -compression_level 6 -pix_fmt yuva420p fire_5_yj_jy.webp
# ffmpeg -i fire_6_yj_jy.mov frames/frame_%04d.png

import os

import cv2
import numpy as np

def image_white(img):
    # img = cv2.imread("img.jpg").astype(np.float32)

    y = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]

    white = np.percentile(y, 95)

    scale = 255.0 / white
    out = np.clip(img * scale, 0, 255)

    # cv2.imwrite("out.jpg", out.astype(np.uint8))
    return out

def video_white(video_path, video_out_path):
    # read .mov video and use image_white() to resave as video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = image_white(frame).astype(np.uint8)
        out.write(frame)
    out.release()
    cap.release()

def create_black_video_basic(output_path='black_video.mp4', duration=3, fps=30, resolution=(640, 480)):
    """
    创建指定时长的全黑视频
    :param output_path: 输出文件路径
    :param duration: 视频时长（秒）
    :param fps: 帧率
    :param resolution: 分辨率 (宽, 高)
    """
    width, height = resolution

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration * fps)

    print(f"正在生成 {duration} 秒全黑视频...")
    print(f"参数: {width}x{height}, {fps}fps, 总帧数: {total_frames}")

    for i in range(total_frames):
        out.write(black_frame)

        if i % 30 == 0:
            progress = (i + 1) / total_frames * 100
            print(f"进度: {progress:.1f}%")

    out.release()
    print(f"视频生成完成: {output_path}")

    return output_path


if __name__ == "__main__":
    # video_path = "/Users/bigo10295/Downloads/gift_exp/reaction.mov"
    # video_out_path = "/Users/bigo10295/Downloads/gift_exp/reaction_white.mov"
    # video_white(video_path, video_out_path)

    create_black_video_basic('black_video.mp4', duration=3, fps=30, resolution=(720, 1280))
