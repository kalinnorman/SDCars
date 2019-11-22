'''
driving with yolo
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
from yolo_4_autumn import Yolo

# # defines
lower_red = np.array([30,150,200])      # I need to test these values
upper_red = np.array([255,255,255])
lower_green = np.array([30,150,50])
upper_green = np.array([255,255,180])
lower_yellow = np.array([30,150,179])
upper_yellow = np.array([255,255,180])
yolo_map = cv2.imread('Maps/yolo_regions.bmp')
yolo_region = 123
yo = Yolo()
img_middle = 208    # this is the middle of the yolo picture, the width is always 416 pixels
yolo_frame_count = 0    # we use this so that we aren't checking yolo at every frame; probably should put this in Sensors.py
color_detected = 255  # used to detect white in the mask pictures

# this function is already in the driving code
def get_gray_value(coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
    imgWidth = img.shape[1] # Get width
    x = round(coordinates[0]) # x translates directly
    y = imgWidth - round(coordinates[1]) # y is inverted
    cur_gps = (x,y)
    gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
    cur_gray_val = gray_val
    return gray_val


# driving stuff

# before anything else, check that we're in the region for yolo
# if we are, check for yolo (have a counter for each frame so you aren't checking
# it at every frame)

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
        bounding_boxes, yolo_img = yo.main_yolo(imgLight)
        light_boxes = []
        # bounding_box = [x1, y1, x2, y2]   # format of bounding_boxes[i]
        for box in range(0, len(bounding_boxes)):
            # x1 = bounding_boxes[box][0]
            # x2 = bounding_boxes[box][2]

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

        imgHSV = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        maskRed = cv2.inRange(imgHSV, lower_red, upper_red)
        maskGreen = cv2.inRange(imgHSV, lower_green, upper_green)
        maskYellow = cv2.inRange(imgHSV, lower_yellow, upper_yellow)

        red_detection = 0
        yellow_detection = 0
        green_detection = 0

        # parse through masked images and check for white
        # whichever mask has the highest amount of white is the winner
        for y in range(0, y2-y1) :
            for x in range(0, x2-x1) :
                if maskRed[y][x] == color_detected :
                    red_detection += 1
                if maskYellow[y][x] == color_detected :
                    yellow_detection += 1
                if maskGreen[y][x] == color_detected :
                    green_detection += 1
        if red_detection > green_detection and red_detection > yellow_detection :
            detection = 'red'
        elif yellow_detection > green_detection :
            detection = 'yellow'
        else :
            detection = 'green'

        print(detection, " is the winner!")
        cv2.imshow("cropped", imgHSV)
        cv2.imshow('red mask',maskRed)
        cv2.imshow('yellow mask',maskYellow)
        cv2.imshow('green mask',maskGreen)

        cv2.waitKey(0)

        # return detection
