import cv2
import numpy as np
import time
from CarControl import CarControl
from datetime import datetime

def resize_img(img):
    scale = 0.5
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dim = (width,height)
    resized = cv2.resize(img, dim)
    cv2.imshow("Plot Route",resized)
    cv2.waitKey(1)

def get_gray_value(coordinates, img):
    imgWidth = img.shape[1]
    x = round(coordinates[0])
    y = imgWidth - round(coordinates[1])
    gray_val = img[x,y]
    return gray_val, x, y

def get_steer_angle(gray_val, prev_gray_val, wr):
    desired_gray_val = 210
    kp = -0.2
    kd = 0.5
    angle = round(kp * (desired_gray_val - gray_val) + kd * (gray_val - prev_gray_val))
    wr_str = "Kp Angle: " + str(kp * (desired_gray_val - gray_val)) + " || Kd Angle: " + str(kd * (gray_val - prev_gray_val))
    wr.write(wr_str + "\n")
    if abs(angle) > 30:
        angle = np.sign(angle)*30
    elif angle == -0.0:
        angle = 0
    return angle

def get_steer_angle_straight_region(gray_val):
    desired_gray_val = 205
    angle = round(gray_val - desired_gray_val)
    # if abs(angle) > 30:
    #     angle = 7 * np.sign(angle)
    if abs(angle) > 15:
        angle = 5 * np.sign(angle)
    elif abs(angle) > 7:
        angle = 3 * np.sign(angle)
    else:
        angle = 1 * np.sign(angle)
    return angle

cc = CarControl()
raw_img = cv2.imread('grayscale_blur.bmp') # 1024 X 1600, ([height],[width]) (0,0) in upper left corner
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
# regions = cv2.imread('straight_regions2.bmp')
# regions = cv2.cvtColor(regions, cv2.COLOR_BGR2GRAY)
speed = 0.27
cc.steer(0)
cc.drive(0.6)
time.sleep(0.3)
cc.drive(speed)

filename = datetime.now().strftime("%b-%d-%Y_%H:%M:%S") + ".txt"
wr = open("LogFiles/"+filename,"w")

prev_gray_val = 1000
went_outside_gps = False

try:
    while True:
        car_location = cc.sensor.get_gps_coord("Blue") # ([height],[width]) (0,0) in upper right corner
        # print(cc.sensor.get_gps_coord("Blue"))
        if car_location[0] > 0:
            gray_val, x, y = get_gray_value(car_location, img) # 205, 250, and 170 are center of lane, center of road, and leaving the road
            raw_img[x,y] = (255,0,0)
            resize_img(raw_img)
            # region_val = get_gray_value(car_location, regions) # 77, 128, 255 are straight road, intersection, and curved road

            # if region_val == 77:
            #     angle = get_steer_angle_straight_region(gray_val)
            # # elif region_val == 128 and car_location[0] > 300 and car_location[0] < 1300:
            #     """Intersection Behavior"""
            # else:
            #     angle = get_steer_angle(gray_val)
            # if went_outside_gps:
            #     angle = -1 * angle
            #     went_outside_gps = False
            #     # FIXME need some sort of handling to check when it should stop negating the angle value (track grey values, when grey values go from increasing to decreasing, or vice versa?)
            if prev_gray_val == 1000:
                prev_gray_val = gray_val
            wr.write("GPS: " + str(car_location) + " || ")
            angle = get_steer_angle(gray_val, prev_gray_val, wr)
            prev_gray_val = gray_val
            cc.steer(angle)
        else:
            went_outside_gps = True
            prev_gray_val = 1000
        
except KeyboardInterrupt:
    wr.close()
    print('Closing program')

