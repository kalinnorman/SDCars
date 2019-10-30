import cv2
import numpy as np
import time
from CarControl import CarControl

cc = CarControl()
img = cv2.imread('grayscale_blur.bmp') # 1024 X 1600, ([height],[width]) (0,0) in upper left corner
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
speed = 0.3
cc.steer(0)
# cc.drive(0.6)
# time.sleep(0.3)
# cc.drive(speed)

try:
    while True:
        car_location = cc.sensor.get_gps_coord("Blue") # ([height],[width]) (0,0) in upper right corner
        print(car_location)
        # if 
        # gray_val = get_gray_value(car_location, img))

        # Make steering decision
        
except:
    print('Closing program')

def get_gray_value(self, coordinates, img):
    imgWidth = img.shape[1]
    x = coordinates[0]
    y = imgWidth - coordinates[1]
    gray_val,_,_ = img[x,y]
    return gray_val