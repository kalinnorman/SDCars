"""
    Main file for driving the car.

    Features:
        - Predictive driving
        - Obstacle detection
        - Waypoint GUI

    14 Nov 2019
"""

from predictiveFollower import PredictiveFollower
from CarControl import CarControl
from datetime import datetime
from drive import Drive

import GlobalParameters as gp
import numpy as np
import queue
import time
import math
import cv2
import sys


# Setup
car = Drive()  # initialize the car
car.cc.steer(0)  # set the steering to straight

# Initialize State information
cur_img = car.lane_follow_img  # The map that we are referencing.
car_location = car.cc.sensor.get_gps_coord("Blue")  # get the GPS coordinates
prev_gps = car_location # Initialize previous GPS to the current GPS
cur_region = car.get_region(car_location)  # indicate where we are
desired_region = gp.region_dict['Region 2']  # indicate where we want to go

# Check whether car is in valid position
if cur_region < 1 or cur_region > 4:  # if the car isn't in region 1-4, stop the script
    print("Car location is not in a valid road, cannot run program")
    car.out_file.close()  # close the output file
    sys.exit()  # terminate the script
else:
    car.update_desired_gray_val(cur_region)

#########  GUI #############

img = cv2.imread('Maps/Global.jpg')
cv2.namedWindow('image', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('image', 1000, 700)
cv2.setMouseCallback('image', add_waypoint)

way_pts = open("/home/nvidia/Desktop/class_code/waypoints.txt", "w+")
while(True):
    cv2.imshow('image', img) 
    key = cv2.waitKey(20 & 0xFF)
    if key == 27: # Escape key
        way_pts.close()
        print("Successfully escaping")
        car.update_waypoints()
        break
    elif key == ord('a'):
        print(mouseX, mouseY)

###################################

desired_coordinates, des_x, des_y, desired_region = car.get_next_coordinates()
print("Initial waypoint coordinates: ", des_x, des_y)


restart_car = False 

try:
    while True:

        # Check for objects
        car.cc.update_sensors()
        object_detected, image = car.cc.detector.detect_object() # Search region in front of car for object           

        if (object_detected):
            car.stop_car()
            print("object detected! ")
            restart_car = True # When the object is removed, this tells the car to start again
            continue           # Skip all the remaining steps until the object is gone

        if (restart_car):
            car.start_car()
            restart_car = False 

        while car_location == prev_gps or prev_gps[0] < 0:

            # Include object detection here?





            # Get GPS coordinates
            car_location = car.cc.sensor.get_gps_coord("Blue")  # ([height],[width]) (0,0) in upper right corner
            if prev_gps[0] < 0: # Updates the prev gps if the car was out of bounds but reentered
                prev_gps = car_location

        car_x = car_location[0] # Get x (not as a tuple)
        car_y = car_location[1] # Get y (not as a tuple)
        # Check if GPS found us
        if car_location[0] > 0:  # if the gps found us

            # Plot car position
            cv2.circle(img, (int(car_y),int(car_x)), 3, (0, 200, 0), 3)
            cv2.imshow('image', img) 
            key = cv2.waitKey(1)

            region = car.get_region(car_location)  # update the current region

            # Check where we are vs. where we want to be.
            if cur_region != gp.region_dict['Intersection'] and region == gp.region_dict['Intersection']:  # Entering the intersection
                cur_img, next_region = car.get_intersection_map(cur_region, desired_region)  # use the appropriate map to turn
                car.predict.set_img(cur_img)
                cur_region = gp.region_dict['Intersection']  # indicate we are in the intersection
                car.update_desired_gray_val(next_region)

            elif cur_region == gp.region_dict['Intersection'] and region != gp.region_dict['Intersection']:  # Leaving the intersection
                cur_img = car.lane_follow_img  # go back to the default map
                car.predict.set_img(cur_img)
                cur_region = next_region  # indicate where we ended up

            elif region == cur_region: # Car is in the appropriate region
                if cur_region == desired_region: # Lane is approximately 70 pixels wide
                    dist_from_waypoint = math.sqrt((des_x-car_x)**2 + (des_y-car_y)**2)
                    if dist_from_waypoint < 40:
                        print("Waypoint Reached!")
                        car.waypoints.pop(0) # get rid of the waypoint
                        # check to see if there are more waypoints
                        if len(car.waypoints) == 0:  # if there are no more coordinates
                            print("All waypoints reached!")
                            break  # we're done!
                        else:  # if there are more coordinates
                            # car.cc.drive(0)
                            # time.sleep(1)
                            # car.cc.drive(car.speed)
                            desired_coordinates, des_x, des_y, desired_region = car.get_next_coordinates()  # get the next location and go!
                            print("New waypoint coordinates:", des_x, des_y)
                            print("Current region:", cur_region)
                            print("Desired region:", desired_region)

            steering_angle = car.get_angle(car_location, prev_gps)
            car.cc.steer(steering_angle)
            prev_gps = car_location

        else:  # if the gps didn't find us
            car.cur_region = gp.region_dict['Out of bounds']  # indicate we are out of bounds
            prev_gps = car_location
            
        car.update_log_file()  # update the log file

    car.cc.drive(0)  # stop the car
    print("Terminating Program")
    car.out_file.close()  # close the log file
    
except KeyboardInterrupt:  # if the user Ctrl+C's us
    print("User Terminated Program")
    car.out_file.close()  # close the log file
