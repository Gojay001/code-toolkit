import numpy as np

import orion
import os
import cv2
import pyperclip

import easyocr
easyocr_reader = easyocr.Reader(["en"])


np.set_printoptions(suppress=True)

def resize_1024p(frame):
    frame = cv2.resize(frame, (1024, 1024))
    black_image = np.zeros((1820, 1024, 3), dtype=np.uint8)
    start_y = (1820 - 1024) // 2
    black_image[start_y:start_y + 1024, :] = frame

    return black_image

all_files = orion.path.get_all_files_with_extension("meta_imgs", (".jpg", ".png"))

dataset_files = {}
for file in all_files:
    file_name = orion.path.extract_path_basename_without_extension(file)
    dataset = file_name.split("_")[0]
    if dataset not in dataset_files:
        dataset_files[dataset] = []
    dataset_files[dataset].append(file)

for dataset in dataset_files:
    print(f"dataset: {dataset}, number of files: {len(dataset_files[dataset])}")

def get_random_file_from_dataset(dataset_files, random_mode="average"):
    if random_mode == "even":
        key_list = list(dataset_files.keys())
        key_list.remove("closedeye") #skip closedeye

        dataset = np.random.choice(key_list)
    elif random_mode == "even_weighted":
        weights = []
        for dataset in dataset_files:
            weights.append(len(dataset_files[dataset]))
        # print(np.array(weights)/sum(weights))
        dataset = np.random.choice(list(dataset_files.keys()), p=np.array(weights)/sum(weights))
    elif random_mode == "custom":
        custom_weights = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        dataset = np.random.choice(list(dataset_files.keys()), p=custom_weights)

    return np.random.choice(dataset_files[dataset])


def generate_random_video(image_num, source_image_name):

    video_writer = cv2.VideoWriter("test_imgs/target.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1024, 1820))

    for i in range(image_num):

        target_image = get_random_file_from_dataset(dataset_files, random_mode="custom")
        print(target_image)

        target_image_cv = cv2.imread(target_image)
        target_image_cv = resize_1024p(target_image_cv)

        target_image_name = orion.path.extract_path_basename_without_extension(target_image)
        cv2.putText(target_image_cv, f"{source_image_name}_to_{target_image_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(target_image_cv, f"{i}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

        video_writer.write(target_image_cv)

    video_writer.release()

    # cmd = f'ffmpeg -i test_imgs/target.mp4 -c:v libx264  -y test_imgs/target_convert.mp4'
    # os.system(cmd)

def save_result_image_from_video(video_path):

    pre_result_file_name = None

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        result_image = frame[0:100, 50:1000, :]
        _, result_image_name, _ = easyocr_reader.readtext(result_image)[0]

        if pre_result_file_name is not None and pre_result_file_name == result_image_name:
            continue
        else:
            pre_result_file_name = result_image_name

        print(result_image_name)
        result_image_path = "result_imgs/" + result_image_name + ".png"

        start_y = (1820 - 1024) // 2
        frame = frame[start_y:start_y + 1024, :, :]

        cv2.imwrite(result_image_path, frame)

    video_capture.release()

#---------------------------------------------------------------------

frame_id = 0

while True:
    source_image = get_random_file_from_dataset(dataset_files, random_mode="even")
    orion.path.copy_file(source_image, "test_imgs/source.png")

    source_image_name = orion.path.extract_path_basename_without_extension(source_image)

    generate_random_video(20, source_image_name)

    os.makedirs("result_imgs", exist_ok=True)

    print("source_image:", source_image)

    frame_id += 1

    print(f"frame_id: {frame_id}")
    pyperclip.copy(f"{source_image_name}_{frame_id}")
    pyperclip.copy(f"{source_image_name}_{frame_id}")
    pyperclip.copy(f"{source_image_name}_{frame_id}")

    print("--------------------------------")
    input("Press Enter to continue...")

    # save_result_image_from_video("test_imgs/result.mp4")
    # if os.path.exists("test_imgs/result.mp4"):
    #     os.remove("test_imgs/result.mp4")

