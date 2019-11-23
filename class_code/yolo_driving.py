'''
driving with yolo
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
from YoloTest import Yolo

# # defines
yolo_map = cv2.imread('Maps/yolo_regions.bmp')
yolo_region = 123
yo = Yolo()
img_middle = 208    # this is the middle of the yolo picture, the width is always 416 pixels
yolo_frame_count = 0    # we use this so that we aren't checking yolo at every frame; probably should put this in Sensors.py

# this function is already in the driving code
def get_gray_value(coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
    imgWidth = img.shape[1] # Get width
    x = round(coordinates[0]) # x translates directly
    y = imgWidth - round(coordinates[1]) # y is inverted
    cur_gps = (x,y)
    gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
    cur_gray_val = gray_val
    return gray_val

def find_color(img, color):
    # Convert image to HSV
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the desired colorspace
    if color == 'red':
        lower = np.array([120, 40, 40], dtype='uint8') # was [150, 40, 40]
        upper = np.array([255, 255, 255], dtype='uint8')
    elif color == 'green':
        lower = np.array([50, 40, 40], dtype='uint8')
        upper = np.array([100, 255, 255], dtype='uint8')
    elif color == 'yellow':
        lower = np.array([0, 40, 40], dtype='uint8')
        upper = np.array([50, 255, 255], dtype='uint8')
    else:
        print("Choose a valid color, bro.")

    # Threshold the HSV image to get only the desired color
    mask = cv2.inRange(imghsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    count = cv2.countNonZero(res[:,:,0])
    cv2.imshow('img', res)
    cv2.waitKey(0)

    return res, count  # returns the image and the count of non-zero pixels


def predict_color(img):

    colors = ['red', 'yellow', 'green']
    counts = []

    for color in colors:
        res, count = find_color(img, color)
        counts.append(count)

    return colors[counts.index(max(counts))]  # returns the color as a string


# driving stuff

# before anything else, check that we're in the region for yolo
# if we are, check for yolo (have a counter for each frame so you aren't checking
# it at every frame)

# construct the argument parse and parse the arguments

coordinates = (300,600) # this is for testing purposes; it's in the NE of the yolo regions
if get_gray_value(coordinates, yolo_map)[0] == yolo_region :
    yolo_frame_count = 10 #+= 1

    if yolo_frame_count == 10 :
        yolo_frame_count = 0
        # this is the equivalent to reading in a frame
        imgLight = cv2.imread('traffic_lights/red_light2.png')
        cv2.imshow("light", imgLight)
        cv2.waitKey(0)
        # do yolo
        # get vector of bounding boxes from yolo and check that they're on the right side
        bounding_boxes, yolo_img = yo.main_yolo(imgLight)#, args)
        light_boxes = []
        # bounding_box = [x1, y1, x2, y2]   # format of bounding_boxes[i]
        for box in range(0, len(bounding_boxes)):
            if bounding_boxes[box][0] > img_middle and bounding_boxes[box][2] > img_middle :  # bounding box is on the right side of the camera
                light_boxes.append(bounding_boxes[box])
                print (light_boxes[-1])
        y_of_light = 400 # arbitrary value that is used to compare when there is more than one detected traffic light
        if not light_boxes:
            print("DEBUG: oh no! there aren't any boxes!") # exit frame and try again
        # we only want to look at one light, so if we detect more than one,
        # we will look at the traffic light that is closest to the top of the pic as
        # that one is likely to be the one we want to look at
        elif len(light_boxes) > 1 :
            for i in range(0, len(light_boxes)) :
                top_y = min(light_boxes[i][1], light_boxes[i][3])
                if top_y < y_of_light :
                    y_of_light = top_y
                    desired_light = i
        else :
            desired_light = 0   # there's only one traffic light detected in the desired region
        ################## I need to double check that y = 0 is the top ############

        # crop image:
        x1 = int(light_boxes[desired_light][0])
        y1 = int(light_boxes[desired_light][1])
        x2 = int(light_boxes[desired_light][2])
        y2 = int(light_boxes[desired_light][3])
        cropped_img = yolo_img[y1:y2, x1:x2]

        color_detected = predict_color(cropped_img)
        print(color_detected, " is the winner!")
        cv2.imshow("cropped", cropped_img)

        cv2.waitKey(0)
