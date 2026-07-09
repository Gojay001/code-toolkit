import os

def merge_two_videos(first_video_path, second_video_path, dst_video_path):
    """"merge two videos"""

    cmd = f'ffmpeg -y -i {first_video_path} -i {second_video_path} -filter_complex hstack=inputs=2 -c:v libx264 {dst_video_path}'
    os.system(cmd)

def merge_three_videos(first_video_path, second_video_path, third_video_path, dst_video_path):
    """"merge three videos"""

    cmd = f'ffmpeg -y -i {first_video_path} -i {second_video_path} -i {third_video_path} -filter_complex hstack=inputs=3 -c:v libx264 {dst_video_path}'
    os.system(cmd)

def merge_four_videos(first_video_path, second_video_path, third_video_path, fourth_video_path, dst_video_path):
    """"merge four videos"""

    cmd = f'ffmpeg -y -i {first_video_path} -i {second_video_path} -i {third_video_path} -i {fourth_video_path} -filter_complex hstack=inputs=4 -c:v libx264 {dst_video_path}'
    os.system(cmd)

def merge_four_videos_2x2(first_video_path, second_video_path, third_video_path, fourth_video_path, dst_video_path):
    """"merge four videos in 2x2 mode"""

    cmd = f'ffmpeg -y -i {first_video_path} -i {second_video_path} -i {third_video_path} -i {fourth_video_path} \
            -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" -c:v libx264 {dst_video_path}'
    os.system(cmd)

#==============================================================================


if __name__ == '__main__':
    # merge_two_videos('res_video/interp_gt.mp4', 'res_video/interp_pred.mp4', 'res_video/interp_merge.mp4')
    merge_two_videos('/Users/bigo10295/Documents/face_attr_data/bvt_res/res_out_1.mp4',
                     '/Users/bigo10295/Documents/face_attr_data/bvt_res/res_out_2.mp4',
                     '/Users/bigo10295/Documents/face_attr_data/bvt_res/res_out_merge.mp4')
    # merge_four_videos('/Users/bigo10295/Downloads/temp_img/3.mp4',
    #                  '/Users/bigo10295/Downloads/temp_img/ghost-gfpgan.mp4',
    #                  '/Users/bigo10295/Downloads/temp_img/deep-live-cam.mp4',
    #                  '/Users/bigo10295/Downloads/temp_img/xs.mp4',
    #                  '/Users/bigo10295/Downloads/temp_img/lyf_3_merge.mp4')