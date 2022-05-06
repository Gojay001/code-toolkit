import os
import shutil
import random


def divide_data(org='../data/org', train='../data/train', val='../data/val', ratio=0.3):
    """
    divide origin files (two-level) into train set and val set
    """
    file_list = [f for f in os.listdir(org) if not f.startswith('.')]

    # mkdir train/ and val/ files
    if not os.path.exists(train):
        os.mkdir(train)
    if not os.path.exists(val):
        os.mkdir(val)

        # join subdirectory in train and val
        for file in file_list:
            train_file = os.path.join(train, file)
            val_file = os.path.join(val, file)
            if not os.path.exists(train_file):
                os.mkdir(train_file)
            if not os.path.exists(val_file):
                os.mkdir(val_file)

            img_list = [f for f in os.listdir(
                os.path.join(org, file)) if not f.startswith('.')]
            random.shuffle(img_list)
            total_img = len(img_list)

            # copy origin images to train and val set
            for idx, img in enumerate(img_list):
                if idx < total_img * ratio:
                    input_img = os.path.join(org, file, img)
                    output_img = os.path.join(val_file, img)
                    shutil.copy(input_img, output_img)
                else:
                    input_img = os.path.join(org, file, img)
                    output_img = os.path.join(train_file, img)
                    shutil.copy(input_img, output_img)

            print(file + "has already copied!")


if __name__ == '__main__':

    divide_data()
