import cv2
import os


def image_resize(img_name, new_size=(1400, 600), file_name='./'):
    img = cv2.imread(file_name + img_name)

    print('Original Dimensions : ', img.shape)

    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    # resized = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

    # vis
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imshow('resized', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(file_name + 'resized_' + img_name, resized)
    print("image has been resized!")


def batch_resize(file_name):
    image_list = os.listdir(file_name)
    # image_list.remove('.DS_Store')
    # image_list.sort(key=lambda x: int(x.split('.')[0]))
    print('image number:', len(image_list))

    for image in image_list:
        image_resize(image, new_size=(200, 200), file_name=file_name)


def hist_equalize(img_name):
    image = cv2.imread(img_name)
    (R, G, B) = cv2.split(image)
    [R, G, B] = [cv2.equalizeHist(channel) for channel in [R, G, B]]
    image = cv2.merge([R, G, B])
    cv2.imwrite('histeq_' + img_name, image)


def gaussian_blur(img_name):
    image = cv2.imread(img_name)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite('gaussblur_' + img_name, image)


if __name__ == '__main__':
    file_name = './imgs/'
    # image_resize('test.jpg', (150, 150))
    batch_resize(file_name)
    # img = 'Porsche1.jpg'
    # hist_equalize(img)
    # gaussian_blur(img)
