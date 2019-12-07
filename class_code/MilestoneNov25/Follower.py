from datetime import datetime
from Planner import Planner
from Control import Control
import Parameters as gp
import numpy as np
import queue
import time
import math
import cv2
import sys
import os



class Follower:
    def __init__(self):
        self.speed = 0.3
        self.angle_multiplier = 0.5
        self.cc = Control(start="1572")
        self.cur_angle = 0
        self.cur_gps = self.cc.sensor.get_gps_coord("Blue")
        self.prev_gps = self.cur_gps
        self.start_time = time.time()
        self.regions_img = cv2.imread('Maps/regions.bmp') # RGB
        self.regions_img = cv2.cvtColor(self.regions_img, cv2.COLOR_BGR2GRAY) # Grayscale
        self.cur_region = self.update_region()
        self.update_desired_region()
        self.intersectionStops = cv2.imread('Maps/IntersectionStops.bmp') # RGB
        self.intersectionStops = cv2.cvtColor(self.intersectionStops, cv2.COLOR_BGR2GRAY) # Grayscale
        self.log_filename = "Log.txt" # Creates file named by the date and time to log items for debugging
        if os.path.exists(self.log_filename):
            os.remove(self.log_filename)
        self.out_file = open(self.log_filename,"w") # Opens (creates the file)
        self.regions1and4 = cv2.imread('Maps/Region1to1_4to4.bmp') # Reads in the RGB image
        self.regions1and4 = cv2.cvtColor(self.regions1and4, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        self.region1to4 = cv2.imread('Maps/Region1to4.bmp') # RGB
        self.region1to4 = cv2.cvtColor(self.region1to4, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region1to3 = cv2.imread('Maps/Region1to3.bmp') # RGB
        self.region1to3 = cv2.cvtColor(self.region1to3, cv2.COLOR_BGR2GRAY) # Grayscale
        self.regions2and3 = cv2.imread('Maps/Region2to3_3to2.bmp') # RGB
        self.regions2and3 = cv2.cvtColor(self.regions2and3, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region2to2 = cv2.imread('Maps/Region2to2.bmp') # RGB
        self.region2to2 = cv2.cvtColor(self.region2to2, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region2to4 = cv2.imread('Maps/Region2to4.bmp') # RGB
        self.region2to4 = cv2.cvtColor(self.region2to4, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region3to3 = cv2.imread('Maps/Region3to3.bmp') # RGB
        self.region3to3 = cv2.cvtColor(self.region3to3, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region3to1 = cv2.imread('Maps/Region3to1.bmp') # RGB
        self.region3to1 = cv2.cvtColor(self.region3to1, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region4to1 = cv2.imread('Maps/Region4to1.bmp') # RGB
        self.region4to1 = cv2.cvtColor(self.region4to1, cv2.COLOR_BGR2GRAY) # Grayscale
        self.region4to2 = cv2.imread('Maps/Region4to2.bmp') # RGB
        self.region4to2 = cv2.cvtColor(self.region4to2, cv2.COLOR_BGR2GRAY) # Grayscale
        
        self.restart_car = False
        self.attempt_time = 2
        
        if self.cur_region == gp.region_dict['Region 1'] or self.cur_region == gp.region_dict['Region 4']:
            self.predict = Planner(self.regions1and4, search_radius=50)
            print("using regions 1 and 4")
        else:
            self.predict = Planner(self.regions2and3, search_radius=50)
            print("using regions 2 and 3")

    def get_angle(self):
        angle_rads = self.predict.find_angle(1600-self.cur_gps[1],self.cur_gps[0],1600-self.prev_gps[1],self.prev_gps[0])
        angle_deg = angle_rads*180.0/np.pi # Convert angle from radians to degrees
        angle_mod = round(self.angle_multiplier*angle_deg)
        if abs(angle_mod) > 30: # Cap the angle at -30 and 30
            angle_mod = np.sign(angle_mod)*30
        self.cur_angle = angle_mod # update the class value tracking the current angle
        return angle_mod
    
    def get_gray_value(self, coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
        imgWidth = img.shape[1] # Get width
        x = round(coordinates[0]) # x translates directly
        y = imgWidth - round(coordinates[1]) # y is inverted
        # self.cur_gps = (x,y)
        gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
        self.cur_gray_val = gray_val
        return gray_val

    def get_intersection_map(self, prev_region):
        # Determine what action to take based on the current region and the region of the desired GPS coordinates
        # Returns the appropriate map for the action to take, and the next region the car will be in
        if prev_region == 1:
            if self.desired_region == 1:
                return self.regions1and4, 1
            elif self.desired_region == 2:
                return self.region1to3, 3
            elif self.desired_region == 3:
                return self.region1to3, 3
            elif self.desired_region == 4:
                return self.region1to4, 4
            else:
                return self.region1to4, 0
        elif prev_region == 2:
            if self.desired_region == 1:
                return self.region2to4, 4
            elif self.desired_region == 2:
                return self.region2to2, 2
            elif self.desired_region == 3:
                return self.regions2and3, 3
            elif self.desired_region == 4:
                return self.region2to4, 4
            else:
                return self.region2to4, 0
        elif prev_region == 3:
            if self.desired_region == 1:
                return self.region3to1, 1
            elif self.desired_region == 2:
                return self.regions2and3, 2
            elif self.desired_region == 3:
                return self.region3to3, 3
            elif self.desired_region == 4:
                return self.region3to1, 1
            else:
                return self.region3to1, 0
        elif prev_region == 4:
            if self.desired_region == 1:
                return self.region4to1, 1
            elif self.desired_region == 2:
                return self.region4to2, 2
            elif self.desired_region == 3:
                return self.region4to2, 2
            elif self.desired_region == 4:
                return self.regions1and4, 4
            else:
                return self.region4to2, 0
        else:
            return self.regions2and3, 0

    def update_region(self):
        current_gray_val = self.get_gray_value(self.cur_gps, self.regions_img)
        self.cur_region = gp.region_dict[gp.region_values[current_gray_val]]
        return self.cur_region

    def update_gps_pos(self, coordinates):
        self.cur_gps = coordinates

    def update_log_file(self):
        if self.cur_region == 0:
            out_string = "OUTSIDE OF GPS BOUNDS"
        else:
            duration = time.time() - self.start_time
            self.start_time = time.time()
            out_string = "Region:"+str(self.cur_region)+" | GPS:"+str(self.cur_gps)+" | Angle:"+str(self.cur_angle)+" | Time Duration:"+str(duration)
        self.out_file.write(out_string + "\n")

    def update_desired_region(self):
        if self.cur_region == gp.region_dict['Region 1']:
            self.desired_region = gp.region_dict['Region 2']
        elif self.cur_region == gp.region_dict['Region 2']:
            self.desired_region = gp.region_dict['Region 3']
        elif self.cur_region == gp.region_dict['Region 3']:
            self.desired_region = gp.region_dict['Region 4']
        else:
            self.desired_region = gp.region_dict['Region 1']
        return self.desired_region

    def check_stop_signs(self):
        if self.cur_region == gp.region_dict['Region 1']:
            if self.cur_gps[0] < 280:
                if self.prev_gps[1] > 230 and self.cur_gps[1] <= 230:
                    return True
        elif self.cur_region == gp.region_dict['Region 2']:
            if self.cur_gps[1] < 75:
                if self.prev_gps[0] > 370 and self.cur_gps[0] <= 370:
                    return True
        elif self.cur_region == gp.region_dict['Region 3']:
            if self.cur_gps[1] > 1500:
                if self.prev_gps[0] < 690 and self.cur_gps[0] >= 690:
                    return True
        elif self.cur_region == gp.region_dict['Region 4']:
            if self.cur_gps[0] > 790:
                if self.prev_gps[1] < 1330 and self.cur_gps[1] >= 1330:
                    return True
        return False

    def start_car(self):
        self.cc.steer(0)
        self.cc.drive(self.speed)
        self.restart_car = False

    def stop_car(self):
        self.cc.steer(0)
        self.cc.drive(0)
        self.restart_car = True
        self.StopTime = time.time()

    def attempt_correction(self):
        if (self.cur_region == gp.region_dict['Region 1'] or self.cur_region == gp.region_dict['Region 4']): # In inside lanes, swerve right
            self.cc.steer(18)
            self.cc.drive(self.speed)
            self.restart_car = False
            print("Swerve right...")
        
        else:  # In outside lanes, swerve left
            self.cc.steer(-18)
            self.cc.drive(self.speed)
            self.restart_car = False
            print("Swerve left...")
        time.sleep(1)
