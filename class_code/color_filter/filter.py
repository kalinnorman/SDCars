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

def find_color_split(img, show_masks=False):

    top = img[0:int(img.shape[0] / 2), :]
    bottom = img[int(img.shape[0] / 2):img.shape[0], :]

    ret, redmask = cv2.threshold(cv2.extractChannel(top, 2), 170, 255, cv2.THRESH_BINARY)  # may need to be 0 on Jetson.
    ret, greenmask = cv2.threshold(cv2.extractChannel(bottom, 1), 127, 255, cv2.THRESH_BINARY)

    redcount = cv2.countNonZero(redmask)
    greencount = cv2.countNonZero(greenmask)

    count = [redcount, greencount]

    if greencount > redcount:
        res = 'green'
    else:
        res = 'red'

    if show_masks:
        cv2.imshow("r", redmask)
        cv2.imshow("g", greenmask)
        cv2.waitKey(0)

    return res, count  # returns the image and the count of non-zero pixels


folder = 'actual_images\\'
# filenames = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg', 'sample5.jpg']
filenames = ['red_1.png', 'red_2.png', 'yellow_1.png', 'green_1.png', 'green_2.png']

for filename in filenames:
    img = cv2.imread(folder + filename)
    color, count = find_color_split(img, show_masks=True)
    print(color, count)