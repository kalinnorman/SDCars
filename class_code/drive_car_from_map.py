import cv2
import numpy as np
import time
from CarControl import CarControl

def get_gray_value(coordinates, img):
    imgWidth = img.shape[1]
    x = round(coordinates[0])
    y = imgWidth - round(coordinates[1])
    gray_val = img[x,y]
    return gray_val

def get_steer_angle(gray_val):
    desired_gray_val = 205
    turn_factor = 0.5
    angle = round((gray_val - desired_gray_val) * turn_factor)
    if abs(angle) > 30:
        angle = np.sign(angle)*angle
    return angle

def get_steer_angle_straight_region(gray_val):
    desired_gray_val = 205
    angle = round(gray_val - desired_gray_val)
    if abs(angle) > 15:
        angle = 3 * np.sign(angle)
    elif abs(angle) > 7:
        angle = 2 * np.sign(angle)
    else:
        angle = 1 * np.sign(angle)
    return angle

cc = CarControl()
img = cv2.imread('grayscale_blur.bmp') # 1024 X 1600, ([height],[width]) (0,0) in upper left corner
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions = cv2.imread('straight_regions.bmp')
regions = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
speed = 0.3
cc.steer(0)
cc.drive(0.6)
time.sleep(0.3)
cc.drive(speed)

try:
    while True:
        car_location = cc.sensor.get_gps_coord("Blue") # ([height],[width]) (0,0) in upper right corner
        # print(cc.sensor.get_gps_coord("Blue"))
        if car_location[0] > 0:
            gray_val = get_gray_value(car_location, img) # 205, 250, and 170 are center of lane, center of road, and leaving the road
            region_val = get_gray_value(car_location, regions) # 77, 128, 255 are straight road, intersection, and curved road

            if region_val == 77:
                angle = get_steer_angle_straight_region(gray_val)
            # elif region_val == 128 and car_location[0] > 300 and car_location[0] < 1300:
                """Intersection Behavior"""
            else:
                angle = get_steer_angle(gray_val)
            cc.steer(angle)
        
except KeyboardInterrupt:
    print('Closing program')

