from CarControl import CarControl
import GlobalParameters as gp
from datetime import datetime
import numpy as np
import queue
import time
import cv2
import sys


class Drive:
    def __init__(self):
        self.speed = 0.35
        self.cc = CarControl()
        self.cur_angle = 0
        self.cur_gps = (0, 0)
        self.cur_gray_val = 0
        self.cur_region = 0
        self.log_filename = datetime.now().strftime("%b-%d-%Y_%H:%M:%S") + ".txt" # Creates file named by the date and time to log items for debugging
        self.out_file = open("LogFiles/"+self.log_filename,"w") # Opens (creates the file)
        self.kp = -0.5 # Kp value for Proportional Control
        self.kd = 2.0 # Kd value for Derivative Control
        self.prev_gray_vals = queue.Queue(7) # Creates a queue to provide a delay for the previous gray value (used in derivative control)
        self.gray_desired = 210 # The gray value that we want the car to follow
        self.lane_follow_img = cv2.imread('Maps/grayscale_blur.bmp') # Reads in the RGB image
        self.lane_follow_img = cv2.cvtColor(self.lane_follow_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        self.recognize_intersection_img = cv2.imread('Maps/map_with_limits.bmp') # RGB
        self.recognize_intersection_img = cv2.cvtColor(self.recognize_intersection_img, cv2.COLOR_BGR2GRAY) # Grayscale
        self.regions_img = cv2.imread('Maps/regions.bmp') # RGB
        self.regions_img = cv2.cvtColor(self.regions_img, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_left = cv2.imread('Maps/intersection_1_left.bmp') # RGB
        self.one_left = cv2.cvtColor(self.one_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_right = cv2.imread('Maps/intersection_1_right.bmp') # RGB
        self.one_right = cv2.cvtColor(self.one_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_straight = cv2.imread('Maps/intersection_1_straight') # RGB
        self.one_straight = cv2.cvtColor(self.one_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_left = cv2.imread('Maps/intersection_2_left.bmp') # RGB
        self.two_left = cv2.cvtColor(self.two_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_right = cv2.imread('Maps/intersection_2_right.bmp') # RGB
        self.two_right = cv2.cvtColor(self.two_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_straight = cv2.imread('Maps/intersection_2_straight') # RGB
        self.two_straight = cv2.cvtColor(self.two_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_left = cv2.imread('Maps/intersection_3_left.bmp') # RGB
        self.three_left = cv2.cvtColor(self.three_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_right = cv2.imread('Maps/intersection_3_right.bmp') # RGB
        self.three_right = cv2.cvtColor(self.three_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_straight = cv2.imread('Maps/intersection_3_straight') # RGB
        self.three_straight = cv2.cvtColor(self.three_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_left = cv2.imread('Maps/intersection_4_left.bmp') # RGB
        self.four_left = cv2.cvtColor(self.four_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_right = cv2.imread('Maps/intersection_4_right.bmp') # RGB
        self.four_right = cv2.cvtColor(self.four_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_straight = cv2.imread('Maps/intersection_4_straight') # RGB
        self.four_straight = cv2.cvtColor(self.four_right, cv2.COLOR_BGR2GRAY) # Grayscale

    def get_angle(self, current_gray, prev_gray):
        cur = float(current_gray) # Set the values to floats to prevent overflow
        prev = float(current_gray)
        ref = float(self.gray_desired)
        angle = round(self.kp*(ref-cur))+round(self.kd*(cur-prev)) # Calculate the angle
        if abs(angle) > 30: # Cap the angle at -30 and 30
            angle = np.sign(angle)*30
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
                return self.two_right, 2
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
        return gp.region_values[current_gray_val]

    def update_log_file(self):
        if self.cur_region == 0:
            out_string = "OUTSIDE OF GPS BOUNDS"
        else:
            out_string = "Region:"+str(self.cur_region)+" | GPS:"+str(self.cur_gps)+" | Gray:"+str(self.cur_gray_val)+" | Angle:"+str(self.cur_angle)
        self.out_file.write(out_string + "\n")

    def update_queue_and_get_prev_gray_val(self, gray_val):
        if not self.prev_gray_vals.full():
            self.prev_gray_vals.put(gray_val)
            prev_gray = gray_val
        else:
            prev_gray = self.prev_gray_vals.get()
            self.prev_gray_vals.put(gray_val)
        return prev_gray


if __name__ == "__main__":

    # Setup
    car = Drive()  # initialize the car
    car.cc.steer(0)  # set the steering to straight
    car.cc.drive(0.6)  # get the car moving
    time.sleep(0.1)  # ...but only briefly
    car.cc.drive(car.speed)  # get the car moving again

    # Initialize State information
    cur_img = car.lane_follow_img  # The map that we are referencing.
    car_location = car.cc.sensor.get_gps_coord("Blue")  # get the GPS coordinates
    cur_region = car.get_region(car_location)  # indicate where we are
    desired_region = gp.region_dict('Region 1')  # indicate where we want to go

    # Check whether car is in valid position
    if cur_region < 1 or cur_region > 4:  # if the car isn't in region 1-4, stop the script
        print("Car location is not in a valid road, cannot run program")
        car.out_file.close()  # close the output file
        sys.exit()  # terminate the script

    # FIXME Read in file or desired coordinate somehow and uncomment code below
    # desired_region = car.get_region(desired_coordinates) # pass in tuple: (x,y)
    # if desired_region == 0 or desired_region == 5:
    #     print("Desired coordinates ", desired_coordinates, " are not located in a valid location")
    #     car.out_file.close()
    #     return

    try:
        # Start driving!
        while True:

            # Get GPS coordinates
            car_location = car.cc.sensor.get_gps_coord("Blue")  # ([height],[width]) (0,0) in upper right corner

            # Check if GPS found us
            if car_location[0] > 0:  # if the gps found us
                region = car.get_region(car_location)  # update the current region

                # Note: from self.recognize_intersection_img the gray value for entering the intersection is 128
                # Update the regions and control the steering
                #FIXME either need to change the regions map to have the intersection start at the lines, or include a check with the limits map to transition to the intersection stuff

                # Check where we are vs. where we want to be.
                if cur_region != gp.region_dict['Intersection'] and region == gp.region_dict['Intersection']:  # Entering the intersection
                    cur_img, next_region = car.get_intersection_map(cur_region, desired_region)  # use the appropriate map to turn
                    cur_region = gp.region_dict['Intersection']  # indicate we are in the intersection
                    gray_val = car.get_gray_value(car_location, cur_img)  # update the current gray value
                    car.cc.steer(car.get_angle(gray_val, car.update_queue_and_get_prev_gray_val(gray_val)))  # ...and steer appropriately
                elif cur_region == gp.region_dict['Intersection'] and region != gp.region_dict['Intersection']:  # Leaving the intersection
                    cur_img = car.lane_follow_img  # go back to the default map
                    cur_region = next_region  # indicate where we ended up
                    gray_val = car.get_gray_value(car_location, cur_img)  # update the current gray value  # TODO Check. redd added this line
                    car.cc.steer(car.get_angle(gray_val, car.update_queue_and_get_prev_gray_val(gray_val)))  # ...and steer appropriately
                elif region == cur_region: # Car is in the appropriate region
                    car.cc.steer(car.get_angle(gray_val, car.update_queue_and_get_prev_gray_val(gray_val)))
                    break  # exit the loop
                # Do nothing if the car is not in the correct region and is not in the intersection
                # as it should already be correcting itself
                #FIXME check to see if car has reached the desired coordinates, if so, end the program (break from the while loop)

            else:  # if the gps didn't find us
                car.cur_region = gp.region_dict['Out of bounds']  # indicate we are out of bounds

                gray_val = car.get_gray_value((-1*car_location[0], -1*car_location[1]), car.lane_follow_img)
                car.update_queue_and_get_prev_gray_val(gray_val)

            car.update_log_file()  # update the log file

        car.cc.drive(0)  # stop the car
        print("Reached the desired GPS coordinates")
        print("Terminating Program")
        car.out_file.close()  # close the log file
        
    except KeyboardInterrupt:  # if the user Ctrl+C's us
        print("User Terminated Program")
        car.out_file.close()  # close the log file