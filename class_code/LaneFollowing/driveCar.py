from carControl import CarControl
import globalParameters as gp
from datetime import datetime
from follower import Drive
import numpy as np
import queue
import time
import math
import cv2
import sys

# Setup
car = Drive()  # initialize the car
car.cc.steer(0)  # set the steering to straight
prev_region = car.cur_region
next_region = car.desired_region
cur_img = car.predict.map
car.cc.drive(car.speed)
try:
    while True:
        # Get new GPS Coordinate
        while car.cur_gps == car.prev_gps: # Wait for new gps coordinate
            car.update_gps_pos()
            if car.prev_gps[0] < 0: # Updates the prev gps if the car was out of bounds but reentered
                car.prev_gps = car.cur_gps
        # Make steering decisions if a valid GPS coordinate was returned
        if car.cur_gps[0] > 0:
            print("GPS is",car.cur_gps,"Prev GPS",car.prev_gps)
            car.update_region()
            # First case: Entering the intersection from any region
            if prev_region != gp.region_dict['Intersection'] and car.cur_region == gp.region_dict['Intersection']: 
                cur_img, next_region = car.get_intersection_map()  # use the appropriate map to turn
                car.predict.set_img(cur_img)
                prev_region = car.cur_region
                print("Entered Intersection")
            # Second case: Leaving the intersection
            elif prev_region == gp.region_dict['Intersection'] and car.cur_region != gp.region_dict['Intersection']:
                if next_region == gp.region_dict['Region 1'] or next_region == gp.region_dict['Region 4']:
                    cur_img = car.regions1and4 
                else:
                    cur_img = car.regions2and3
                car.predict.set_img(cur_img)
                prev_region = next_region
                print("Left Intersection")
            # Third Case: Car is located in the region it is meant to be in for driving commands
            elif prev_region == car.cur_region:
                if car.cur_region == car.desired_region: # Lane is approximately 70 pixels wide
                    car.update_desired_region()
            steering_angle = car.get_angle()
            car.cc.steer(steering_angle)
            car.prev_gps = car.cur_gps
            print("In the correct region")

        else:  # if the gps didn't find us
            car.cur_region = gp.region_dict['Out of bounds']  # indicate we are out of bounds
            car.prev_gps = car.cur_gps
           
        car.update_log_file()  # update the log file
    
except KeyboardInterrupt:  # if the user Ctrl+C's us
    print("User Terminated Program")
    car.out_file.close()  # close the log file
