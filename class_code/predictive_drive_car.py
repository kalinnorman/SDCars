from predictiveFollower import PredictiveFollower
from CarControl import CarControl
import GlobalParameters as gp
from datetime import datetime
import numpy as np
import queue
import time
import math
import cv2
import sys


class Drive:
    def __init__(self):
        self.speed = 0.35
        self.angle_multiplier = 0.7
        self.cc = CarControl()
        self.predict = PredictiveFollower()
        self.cur_angle = 0
        self.cur_gps = (0, 0)
        self.cur_gray_val = 0
        self.cur_region = 0
        self.log_filename = datetime.now().strftime("%b-%d-%Y_%H:%M:%S") + ".txt" # Creates file named by the date and time to log items for debugging
        self.out_file = open("LogFiles/"+self.log_filename,"w") # Opens (creates the file)
        self.waypoints_filename = "waypoints.txt"
        self.waypoints = []
        self.gray_desired = 220 # 210 # The gray value that we want the car to follow
        self.lane_follow_img = cv2.imread('Maps/grayscale_blur_cutting_corners.bmp') # Reads in the RGB image
        self.lane_follow_img = cv2.cvtColor(self.lane_follow_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        self.regions_img = cv2.imread('Maps/regions.bmp') # RGB
        self.regions_img = cv2.cvtColor(self.regions_img, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_left = cv2.imread('Maps/intersection_1_left.bmp') # RGB
        self.one_left = cv2.cvtColor(self.one_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_right = cv2.imread('Maps/intersection_1_right.bmp') # RGB
        self.one_right = cv2.cvtColor(self.one_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_straight = cv2.imread('Maps/intersection_1_straight.bmp') # RGB
        self.one_straight = cv2.cvtColor(self.one_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_left = cv2.imread('Maps/intersection_2_left.bmp') # RGB
        self.two_left = cv2.cvtColor(self.two_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_right = cv2.imread('Maps/intersection_2_right.bmp') # RGB
        self.two_right = cv2.cvtColor(self.two_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_straight = cv2.imread('Maps/intersection_2_straight.bmp') # RGB
        self.two_straight = cv2.cvtColor(self.two_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_left = cv2.imread('Maps/intersection_3_left.bmp') # RGB
        self.three_left = cv2.cvtColor(self.three_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_right = cv2.imread('Maps/intersection_3_right.bmp') # RGB
        self.three_right = cv2.cvtColor(self.three_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_straight = cv2.imread('Maps/intersection_3_straight.bmp') # RGB
        self.three_straight = cv2.cvtColor(self.three_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_left = cv2.imread('Maps/intersection_4_left.bmp') # RGB
        self.four_left = cv2.cvtColor(self.four_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_right = cv2.imread('Maps/intersection_4_right.bmp') # RGB
        self.four_right = cv2.cvtColor(self.four_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_straight = cv2.imread('Maps/intersection_4_straight.bmp') # RGB
        self.four_straight = cv2.cvtColor(self.four_straight, cv2.COLOR_BGR2GRAY) # Grayscale

        self.get_waypoints()

    def get_angle(self,cur_gps,prev_gps):
        angle_rads = self.predict.find_angle(1600-cur_gps[1],cur_gps[0],1600-prev_gps[1],prev_gps[0])
        angle_deg = angle_rads*180.0/np.pi # Convert angle from radians to degrees
        angle_mod = round(self.angle_multiplier*angle_deg)
        if abs(angle_mod) > 30: # Cap the angle at -30 and 30
            angle = np.sign(angle_mod)*30
        self.cur_angle = angle # update the class value tracking the current angle
        return angle
    
    def get_gray_value(self, coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
        imgWidth = img.shape[1] # Get width
        x = round(coordinates[0]) # x translates directly
        y = imgWidth - round(coordinates[1]) # y is inverted
        self.cur_gps = (x,y)
        gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
        self.cur_gray_val = gray_val
        return gray_val

    def get_intersection_map(self, cur_region, desired_region):
        # Determine what action to take based on the current region and the region of the desired GPS coordinates
        # Returns the appropriate map for the action to take, and the next region the car will be in
        if cur_region == 1:
            if desired_region == 1:
                return self.one_right, 1
            elif desired_region == 2:
                return self.one_straight, 3
            elif desired_region == 3:
                return self.one_straight, 3
            elif desired_region == 4:
                return self.one_left, 4
            else:
                return self.lane_follow_img, 0
        elif cur_region == 2:
            if desired_region == 1:
                return self.two_straight, 4
            elif desired_region == 2:
                return self.two_left, 2
            elif desired_region == 3:
                return self.two_right, 3
            elif desired_region == 4:
                return self.two_straight, 4
            else:
                return self.lane_follow_img, 0
        elif cur_region == 3:
            if desired_region == 1:
                return self.three_straight, 1
            elif desired_region == 2:
                return self.three_right, 2
            elif desired_region == 3:
                return self.three_left, 3
            elif desired_region == 4:
                return self.three_straight, 1
            else:
                return self.lane_follow_img, 0
        elif cur_region == 4:
            if desired_region == 1:
                return self.four_left, 1
            elif desired_region == 2:
                return self.four_straight, 2
            elif desired_region == 3:
                return self.four_straight, 2
            elif desired_region == 4:
                return self.four_right, 4
            else:
                return self.lane_follow_img, 0
        else:
            return self.lane_follow_img, 0

    def get_region(self, coordinates):
        current_gray_val = self.get_gray_value(coordinates, self.regions_img)
        self.cur_region = gp.region_dict[gp.region_values[current_gray_val]]
        return self.cur_region

    def update_log_file(self):
        if self.cur_region == 0:
            out_string = "OUTSIDE OF GPS BOUNDS"
        else:
            out_string = "Region:"+str(self.cur_region)+" | GPS:"+str(self.cur_gps)+" | Gray:"+str(self.cur_gray_val)+" | Angle:"+str(self.cur_angle)
        self.out_file.write(out_string + "\n")

    def get_waypoints(self):
        try:
            waypoints_file = open(self.waypoints_filename,"r") # open the text file with waypoints
            # waypoints = [] # Initialize list to hold all waypoints
            lines = waypoints_file.readlines() # Reads in each line of the file into a list
            for i in lines:
                if not i.__contains__('#'): # If the line is not a comment
                    try:
                        x,y = i.split(',') # split the line at the comma
                        temp_tuple = (int(x),int(y)) # Create a tuple of the gps coordinates
                        self.waypoints.append(temp_tuple) # Add the tuple to the list of waypoints
                    except:
                        pass  # don't do anything
            if len(self.waypoints) == 0:
                print("ERROR: No valid waypoints in file,", self.waypoints_filename)
                print("Terminating program")
                sys.exit()
            # return waypoints
        except:
            print("ERROR: waypoints file:", self.waypoints_filename, "does not exist.")
            print("Terminating program")
            sys.exit()

    def get_next_coordinates(self):
        desired_coordinates = self.waypoints[0]
        des_x = desired_coordinates[0]
        des_y = desired_coordinates[1]
        desired_region = car.get_region(desired_coordinates)  # pass in tuple: (x,y)
        if desired_region == 0 or desired_region == 5:
            print("Desired coordinates", desired_coordinates, "are not located in a valid location")
            car.out_file.close()
            sys.exit()

        return desired_coordinates, des_x, des_y, desired_region


if __name__ == "__main__":

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

    desired_coordinates, des_x, des_y, desired_region = car.get_next_coordinates()
    print("Initial waypoint coordinates: ", des_x, des_y)

    # Begin Driving
    car.cc.drive(0.6)  # get the car moving
    time.sleep(0.1)  # ...but only briefly
    car.cc.drive(car.speed)  # get the car moving again

    try:
        # Start driving!
        while True:
            while car_location == prev_gps or prev_gps[0] < 0:
                # Get GPS coordinates
                car_location = car.cc.sensor.get_gps_coord("Blue")  # ([height],[width]) (0,0) in upper right corner
                if prev_gps[0] < 0: # Updates the prev gps if the car was out of bounds but reentered
                    prev_gps = car_location
            car_x = car_location[0] # Get x (not as a tuple)
            car_y = car_location[1] # Get y (not as a tuple)
            # Check if GPS found us
            if car_location[0] > 0:  # if the gps found us
                region = car.get_region(car_location)  # update the current region

                # Check where we are vs. where we want to be.
                if cur_region != gp.region_dict['Intersection'] and region == gp.region_dict['Intersection']:  # Entering the intersection
                    cur_img, next_region = car.get_intersection_map(cur_region, desired_region)  # use the appropriate map to turn
                    car.predict.set_img(cur_img)
                    cur_region = gp.region_dict['Intersection']  # indicate we are in the intersection
                    if next_region == gp.region_dict['Region 1'] or next_region == gp.region_dict['Region 4']:
                        car.predict.set_gray_val(230)
                    else:
                        car.predict.set_gray_val(215)
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
                                car.cc.drive(0)
                                time.sleep(1)
                                car.cc.drive(speed)
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
