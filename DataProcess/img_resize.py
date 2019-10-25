# -*- coding: utf-8 -*-
import cv2
import numpy as np
img = cv2.imread('../images/test.png')

height, width = img.shape[:2]
print('Original Dimensions : ',img.shape)

# cv.INTER_NEAREST -> to small size
# cv.INTER_CUBIC -> slow
# cv.INTER_LINEAR -> default
# cv.INTER_AREA
resized = cv2.resize(img, (1400, 600), interpolation=cv2.INTER_CUBIC)
# resized = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imshow('resized', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('../images/resized.png', resized)
