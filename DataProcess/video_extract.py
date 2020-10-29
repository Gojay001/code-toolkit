import cv2
import os
import time


path_video = '../video/'
path_save = '../data/'


def to_images(path_video, path_save):
    for root, dirs, files_video in os.walk(path_video):
        print(files_video)

        # read all video files
        for files_name in files_video:
            print(files_name)
            pic_path = path_save + files_name[:-4] + '/img1/'
            os.makedirs(pic_path, exist_ok=True)

            # open video and write to .jpg files        
            video = cv2.VideoCapture(path_video + files_name)
            if (video.isOpened() == False):
                print("error opening video stream or file!")
            
            success = True
            count = 1
            while (video.isOpened()):
                success, frame = video.read()
                if not success:
                    break
                cv2.imwrite(os.path.join(pic_path, '%06d.jpg'%count), frame)
                cv2.waitKey(1)
                count = count+1
                if (count > 2000):
                    break

            # get video info and save to .ini file
            fps =video.get(cv2.CAP_PROP_FPS)
            size = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            ini_path = path_save + files_name[:-4] + '/seqinfo.ini'
            save_ini(files_name[:-4], ini_path, fps, size, count-1)

            video.release()
            print('the current video' +  files_name  + ' is done')


def save_ini(files_name, ini_path, fps, size, length):
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
    to_images(path_video, path_save)
    end = time.time()
    print ('time = %f s\t\n'%(end - start))
    # fps = 15.0
    # size = [100, 200]
    # count = 900
    # ini_path = './data/MOT16-11/seqinfo.ini'
    # save_ini('MOT16-11', ini_path, fps, size, count)
    
    
    
