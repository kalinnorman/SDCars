"""
Color Detection Script

If you run just this script, a demo will show white and yellow detection.

Author: redd
"""

import cv2


def thresholding(img, thresholds, invert):
    """
    Performs thresholding for color detection for each color channel (BRG)
    :param img: The image in which to search
    :param thresholds: The threshold values for a given color
    :param invert: Whether or not the channel must be inverted or not
    :return: The thresholded image
    """
    blue_threshold = thresholds[0]
    red_threshold = thresholds[1]
    green_threshold = thresholds[2]

    blue_invert = invert[0]
    red_invert = invert[1]
    green_invert = invert[2]

    imgb = cv2.threshold(cv2.extractChannel(img, 0), blue_threshold, 255, cv2.THRESH_BINARY)[1]
    imgr = cv2.threshold(cv2.extractChannel(img, 1), red_threshold, 255, cv2.THRESH_BINARY)[1]
    imgg = cv2.threshold(cv2.extractChannel(img, 2), green_threshold, 255, cv2.THRESH_BINARY)[1]

    if blue_invert:
        imgb = cv2.bitwise_not(imgb)

    if red_invert:
        imgr = cv2.bitwise_not(imgr)

    if green_invert:
        imgg = cv2.bitwise_not(imgg)

    return cv2.bitwise_and(imgb, cv2.bitwise_and(imgr, imgg))


def detect_color(img, color):
    """
    Returns a matrix with the given color
    :param img: The image in which to look for a color
    :param color: The color to find
    :return: The image where white indicates that color is in that part of the image.
    """
    if color == "yellow":
        return thresholding(img, (200, 120, 130), (True, False, False))
    elif color == "white":
        return thresholding(img, (200, 120, 130), (False, False, False))
    # elif color == "blue":
    #     return thresholding(img, (120, 200, 200), (False, True, True))


def detect_hue(img, color, sat_thresh=0, val_thresh=0):

    imgval = cv2.extractChannel(img, 2)

    if color == "yellow":
        imghue1 = cv2.threshold(cv2.extractChannel(img, 0), 20, 255, cv2.THRESH_BINARY)[1]
        imghue2 = cv2.threshold(cv2.extractChannel(img, 0), 40, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow('hue1', imghue1)
        #cv2.imshow('hue2', imghue2)
        imghue = cv2.bitwise_and(imghue1, imghue2)
        #cv2.imshow('hue', imghue)
        #cv2.waitKey(0)

        imgsat = cv2.threshold(cv2.extractChannel(img, 1), sat_thresh, 255, cv2.THRESH_BINARY)[1]
        imgval = cv2.threshold(cv2.extractChannel(img, 2), val_thresh, 255, cv2.THRESH_BINARY)[1]
        imgmask = cv2.bitwise_and(imgsat, imgval)
        result = cv2.bitwise_and(imghue, imgmask, dst=None)
        #cv2.imshow('result', result)
        return result
    elif color == "white":
        if val_thresh == 0:
            val_thresh = 140
        if sat_thresh == 0:
            sat_thresh = 150
        imgsat = cv2.threshold(cv2.extractChannel(img, 2), sat_thresh, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow('Saturation', imgsat)
        imgval = cv2.threshold(cv2.extractChannel(img, 2), val_thresh, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow('Value', imgval)
        return cv2.bitwise_and(imgsat, imgval)


if __name__ == "__main__":
    img = cv2.imread('real_straight.jpg')

    cv2.imshow("Original", img)
    cv2.imshow("Yellow", detect_color(img, "yellow"))
    cv2.imshow("White", detect_color(img, "white"))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
