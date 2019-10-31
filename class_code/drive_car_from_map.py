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
    turn_factor = 0.3
    angle = round((gray_val - desired_gray_val) * turn_factor)
    if angle < -30:
        angle = -30
    elif angle > 30:
        angle = 30
    return angle

cc = CarControl()
img = cv2.imread('grayscale_blur.bmp') # 1024 X 1600, ([height],[width]) (0,0) in upper left corner
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions = cv2.imread('straight_regions.bmp')
regions = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
speed = 0.0
cc.steer(0)
cc.drive(0.6)
time.sleep(0.3)
cc.drive(speed)

try:
    while True:
        car_location = cc.sensor.get_gps_coord("Blue") # ([height],[width]) (0,0) in upper right corner
        # print(cc.sensor.get_gps_coord("Blue"))
        if car_location[0] > 0:
            gray_val = get_gray_value(car_location, img)
            region_val = get_gray_value(car_location, regions)
            print(region_val)
            # Gray Val around the center of the lane tends to be around 205
            # Gray val around the center of the road tends to be close to 250
            # Gray val leaving the road tends to be 170 or less (this varies the most)
            angle = get_steer_angle(gray_val)
            cc.steer(angle)
        
except KeyboardInterrupt:
    print('Closing program')

