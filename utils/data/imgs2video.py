import cv2
import glob
import time
import os


def to_video(path_imgs='../data/imgs/', path_save='../video/', video_size=(1920, 1080)):
    """
    convert images files to video.
    """
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = 'saved.avi'
    videoWriter = cv2.VideoWriter(path_save + video_name, fourcc, fps, video_size)

    # read all images
    for img in glob.glob(path_imgs + "*.jpg"):
        print(img)
        frame = cv2.imread(img)
        cv2.imshow("show", frame)
        videoWriter.write(frame)
        # cv2.waitKey(500)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    videoWriter.release()


def to_images(path_video='../video/', path_save='../data'):
    """
    convert videos to image files.
    """
    for root, dirs, files_video in os.walk(path_video):
        print(files_video)

        # read all video files
        for files_name in files_video:
            print(files_name)
            pic_path = path_save + files_name[:-4] + '/imgs/'
            os.makedirs(pic_path, exist_ok=True)

            # open video and write to .jpg files
            video = cv2.VideoCapture(path_video + files_name)
            if not video.isOpened():
                print("error opening video stream or file!")

            count = 1
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                cv2.imwrite(os.path.join(pic_path, '%06d.jpg' % count), frame)
                cv2.waitKey(1)
                count += 1
                if count > 2000:
                    break

            # get video info and save to .ini file
            fps = video.get(cv2.CAP_PROP_FPS)
            size = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            ini_path = path_save + files_name[:-4] + '/seqinfo.ini'
            save_ini(files_name[:-4], ini_path, fps, size, count - 1)

            video.release()
            print('the current video' + files_name + ' is done')


def save_ini(files_name, ini_path, fps, size, length):
    """
    save video information with MOT format.
    """
    data = '[Sequence]\n'
    data += str('name=' + files_name + '\n')
    data += 'imDir=img1\n'
    data += 'frameRate=' + str(fps) + '\n'
    data += 'seqLength=' + str(length) + '\n'
    data += 'imgWidth=' + str(size[0]) + '\n'
    data += 'imgHeight=' + str(size[1]) + '\n'
    data += 'imExt=.jpg\n'
    with open(ini_path, 'w') as f:
        f.write(data)


if __name__ == '__main__':
    start = time.time()
    to_video('../data/imgs/', '../video/', (1920, 1080))
    # to_images('../video/', '../data')

    # fps = 15.0
    # size = [100, 200]
    # count = 900
    # ini_path = '../data/MOT16/seqinfo.ini'
    # save_ini('MOT16', ini_path, fps, size, count)

    end = time.time()
    print('time = {}'.format(end - start))
