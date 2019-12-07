from datetime import datetime
from Follower import Follower
from Control import Control
import Parameters as gp
import numpy as np
import queue
import time
import math
import cv2
import sys

pixel_threshold = 215 # How many pixels constitutes a car in front of us

# Setup
car = Follower()  # initialize the car
prev_region = car.cur_region
next_region = car.desired_region
cur_img = car.predict.map
car.start_car()
object_detected = False
stopAtIntersection = True
# wait = False
# wait_duration = 1.0/5.0
# wait_time_begin = time.time() - wait_duration
try:
    while True:
        # Get new GPS Coordinate and check for objects
        while car.cur_gps == car.prev_gps or object_detected: # Don't move on in the code until the car has moved
            car.update_log_file()  # update the log file
            coords = car.cc.update_sensors()
            car.update_gps_pos(coords)
            car.update_region()
            pixel_count, object_detected, image = car.cc.detector.detect_object()
            if (object_detected):
                if (not car.restart_car): # meaning the car is just stopping
                    car.stop_time = time.time()
                car.stop_car()
                if ( (time.time() - car.stop_time) > car.attempt_time) and (pixel_count < pixel_threshold):  # after a few seconds, try to fix
                    car.attempt_correction()
                continue
            if (car.restart_car):
                car.start_car()
            
            # if car.cur_gps[0] > 1024 or car.cur_gps[1] > 1600:
            #     continue
            # elif car.cur_gps[0] < 0 or car.cur_gps[1] < 0:
            #     continue
#            print(car.get_gray_value(car.cur_gps, car.stops_and_lights)) # FIXME Delete this line once we have the gray value for YOLO
            if car.prev_gps[0] < 0: # Updates the prev gps if the car was out of bounds but reentered
                car.prev_gps = car.cur_gps
            # if time.time()-wait_time_begin >= wait_duration:
            #     wait = False
            # else:
            #     time.sleep(1.0/20.0)
        # wait_time_begin = time.time()
        # wait = True
        # Make steering decisions if a valid GPS coordinate was returned
        if car.cur_gps[0] > 0:
            # car.update_region()
            # First case: Entering the intersection from any region
            if prev_region != gp.region_dict['Intersection'] and car.cur_region == gp.region_dict['Intersection']: 
                cur_img, next_region = car.get_intersection_map(prev_region)  # use the appropriate map to turn
                car.predict.set_img(cur_img)
                prev_region = car.cur_region
                car.cc.detector.inIntersection = True
            # Second case: Leaving the intersection
            elif prev_region == gp.region_dict['Intersection'] and car.cur_region != gp.region_dict['Intersection']:
                if next_region == gp.region_dict['Region 1'] or next_region == gp.region_dict['Region 4']:
                    cur_img = car.regions1and4 
                else:
                    cur_img = car.regions2and3
                car.predict.set_img(cur_img)
                prev_region = next_region
                car.cc.detector.inIntersection = False
            # Third Case: Car is located in the region it is meant to be in for driving commands
            elif prev_region == car.cur_region:
                if car.check_stop_signs(): # If at a stop sign location
                    car.cc.steer(0)
                    car.cc.drive(0)
                    print("Stopped At Stop Sign")
                    time.sleep(2)
                    print("Starting At Stop Sign")
                    car.cc.drive(car.speed)
                    stopAtIntersection = True
                elif car.get_gray_value(car.cur_gps, car.intersectionStops) == 128 and stopAtIntersection:
                    car.cc.steer(0)
                    car.cc.drive(0)
                    stopAtIntersection = False
                    while not car.cc.sensor.green_light:
                        car.cc.update_sensors(yolo_flag=True)
                    car.cc.drive(car.speed)
                    car.cc.sensor.green_light = False
                elif car.cur_region == car.desired_region:
                    car.update_desired_region()
            steering_angle = car.get_angle()
            car.cc.steer(steering_angle)
            car.prev_gps = car.cur_gps

        else:  # if the gps didn't find us
            car.cur_region = gp.region_dict['Out of bounds']  # indicate we are out of bounds
            car.prev_gps = car.cur_gps
    
except KeyboardInterrupt:  # if the user Ctrl+C's us
    print("User Terminated Program")
    car.out_file.close()  # close the log file
