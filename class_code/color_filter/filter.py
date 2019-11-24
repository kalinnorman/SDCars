import cv2
import numpy as np


def find_color(img):

    ret, redmask = cv2.threshold(cv2.extractChannel(img, 2), 127, 255, cv2.THRESH_BINARY)  # may need to be 0 on Jetson.
    ret, greenmask = cv2.threshold(cv2.extractChannel(img, 1), 127, 255, cv2.THRESH_BINARY)

    redcount = cv2.countNonZero(redmask)
    greencount = cv2.countNonZero(greenmask)

    count = [redcount, greencount]

    pixel_threshold = int(0.1 * img.shape[0] * img.shape[1])

    if greencount > pixel_threshold and redcount < pixel_threshold:
        res = 'green'
    else:
        res = 'red'

    return res, count  # returns the image and the count of non-zero pixels


filenames = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg', 'sample5.jpg']

for filename in filenames:
    img = cv2.imread(filename)
    color, count = find_color(img)
    print(color, count)