import cv2
import glob
import time


path_imgs = '../data/imgs/'
path_save = '../video/'
video_name = 'saved.avi'


def to_video(path_imgs, path_save, video_size):
	fps = 24
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
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


if __name__ == '__main__':
    start = time.time()
    video_size = (1920,1080)
    to_video(path_imgs, path_save, video_size)
    end = time.time()
    print ('time = %f s\t\n'%(end - start))