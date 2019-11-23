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
# YOLO
from matplotlib import pyplot as plt
from gluoncv import model_zoo, utils
import pyrealsense2 as rs
from PIL import Image
from signal import signal, SIGINT
from sys import exit
import mxnet as mx
import argparse
import imutils
#import serial
import os
import gc

# before running code, type in su and enter password to be in sudo mode

class Drive:
    def __init__(self):
        self.speed = 0.35
        self.angle_multiplier = 0.7
        self.cc = CarControl()
        self.cur_angle = 0
        self.cur_gps = (0, 0)
        self.cur_gray_val = 0
        self.cur_region = 0
        self.log_filename = datetime.now().strftime("%b-%d-%Y_%H:%M:%S") + ".txt" # Creates file named by the date and time to log items for debugging
        self.out_file = open("LogFiles/"+self.log_filename,"w") # Opens (creates the file)
        self.waypoints_filename = "waypoints.txt"
        self.waypoints = []
        self.gray_desired = 220 # 210 # The gray value that we want the car to follow
        self.lane_follow_img = cv2.imread('Maps/binary_drivable_rounded_outside_expanded_inside_2_blurred.bmp') # Reads in the RGB image
        self.lane_follow_img = cv2.cvtColor(self.lane_follow_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        self.regions_img = cv2.imread('Maps/regions.bmp') # RGB
        self.regions_img = cv2.cvtColor(self.regions_img, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_left = cv2.imread('Maps/3to2_and_1to4.bmp') # RGB
        self.one_left = self.three_right = cv2.cvtColor(self.one_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_right = cv2.imread('Maps/1to1_and_2to2.bmp') # RGB
        self.one_right = self.two_left = cv2.cvtColor(self.one_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_straight = cv2.imread('Maps/intersection_1_straight.bmp') # RGB
        self.one_straight = cv2.cvtColor(self.one_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_right = cv2.imread('Maps/2to3_and_4to1.bmp') # RGB
        self.two_right = self.four_left = cv2.cvtColor(self.two_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_straight = cv2.imread('Maps/intersection_2_straight.bmp') # RGB
        self.two_straight = cv2.cvtColor(self.two_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_left = cv2.imread('Maps/3to3_and_4to4.bmp') # RGB
        self.three_left = self.four_right = cv2.cvtColor(self.three_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_straight = cv2.imread('Maps/intersection_3_straight.bmp') # RGB
        self.three_straight = cv2.cvtColor(self.three_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_straight = cv2.imread('Maps/intersection_4_straight.bmp') # RGB
        self.four_straight = cv2.cvtColor(self.four_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.predict = PredictiveFollower(self.lane_follow_img, search_radius=50)
        self.get_waypoints()

        # YOLO
        self.yolo_map = cv2.imread('Maps/yolo_regions.bmp')
        self.yolo_region_color = 123
        self.yolo_region = False
        self.img_middle = 208    # this is the middle of the yolo picture, the width is always 416 pixels
        self.yolo_frame_count = 0    # we use this so that we aren't checking yolo at every frame; probably should put this in Sensors.py
        #self.yo = Yolo()
        self.green_light = False

        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
        self.args = vars(self.ap.parse_args())

        (self.W, self.H) = (None, None)

        # Implement YOLOv3MXNet
        self.net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

        # Set device to GPU
        self.device=mx.gpu()
        # device=mx.cpu()

        self.net.collect_params().reset_ctx(self.device)

        signal(SIGINT, self.handler)
        print('Running. Press CTRL-C to exit')

        #os.system('MXNET_CUDNN_AUTOTUNE_DEFAULT=0')

    # YOLO
    def find_color(self, img, color):
        # Convert image to HSV
        imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the desired colorspace
        if color == 'red':
            lower = np.array([120, 40, 40], dtype='uint8') # was [150, 40, 40]
            upper = np.array([255, 255, 255], dtype='uint8')
        elif color == 'green':
            lower = np.array([50, 40, 40], dtype='uint8')
            upper = np.array([100, 255, 255], dtype='uint8')
        elif color == 'yellow':
            lower = np.array([0, 40, 40], dtype='uint8')
            upper = np.array([50, 255, 255], dtype='uint8')
        else:
            print("Choose a valid color, bro.")

        # Threshold the HSV image to get only the desired color
        mask = cv2.inRange(imghsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)
        count = cv2.countNonZero(res[:,:,0])
        cv2.imshow('img', res)
        cv2.waitKey(0)

        return res, count  # returns the image and the count of non-zero pixels

    # YOLO
    def predict_color(self, img):

        colors = ['red', 'yellow', 'green']
        counts = []

        for color in colors:
            res, count = self.find_color(img, color)
            counts.append(count)

        return colors[counts.index(max(counts))]  # returns the color as a string

    def get_angle(self,cur_gps,prev_gps):
        angle_rads = self.predict.find_angle(1600-cur_gps[1],cur_gps[0],1600-prev_gps[1],prev_gps[0])
        angle_deg = angle_rads*180.0/np.pi # Convert angle from radians to degrees
        angle_mod = round(self.angle_multiplier*angle_deg)
        if abs(angle_mod) > 30: # Cap the angle at -30 and 30
            angle_mod = np.sign(angle_mod)*30
        self.cur_angle = angle_mod # update the class value tracking the current angle
        return angle_mod

    """Transforms for YOLO series."""
    def transform_test(self, imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if isinstance(imgs, mx.nd.NDArray):
            imgs = [imgs]
        for im in imgs:
            assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

        tensors = []
        origs = []
        for img in imgs:
            orig_img = img.asnumpy().astype('uint8')
            img = mx.nd.image.to_tensor(img)

            img = mx.nd.image.normalize(img, mean=mean, std=std)

            tensors.append(img.expand_dims(0))
            origs.append(orig_img)
        if len(tensors) == 1:
            return tensors[0], origs[0]
        return tensors, origs

    def load_test(self, filenames, short=416):
        if not isinstance(filenames, list):
            filenames = [filenames]
        imgs = [self.letterbox_image(f, short) for f in filenames]
        return self.transform_test(imgs)


    # this function is from yolo3.utils.letterbox_image
    def letterbox_image(self, image, size=416):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size

        scale = min(size/iw, size/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (size, size), (128, 128, 128))
        new_image.paste(image, ((size-nw)//2, (size-nh)//2))
        return mx.nd.array(np.array(new_image))

    # Function to correctly exit program
    def handler(self, signal_received, frame):
        vs.release()
        cv2.destroyAllWindows()
        print('CTRL-C detected. Exiting gracefully')
        exit(0)

    def main_yolo(self, vs) :

        current_bb = [0, 0, 0, 0]
        frame = vs
        print("DEBUG: I'm in YOLO!")

        # if the frame dimensions are empty, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # from gluoncv import data
        yolo_image = Image.fromarray(frame, 'RGB')
        x, img = self.load_test(yolo_image, short=416)

        class_IDs, scores, bounding_boxs = self.net(x.copyto(self.device))

        # The next two lines draw boxes around detected objects
        ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=self.net.classes)
        plt.show()

        # Convert to numpy arrays, then to lists
        class_IDs = class_IDs.asnumpy().tolist()
        scores = scores.asnumpy().tolist()
        bounding_boxs = bounding_boxs.asnumpy()
        traffic_boxes = []
        # iterate through detected objects
        for i in range(len(class_IDs[0])):
            if ((scores[0][i])[0]) > self.args["confidence"]:
                current_class_id = self.net.classes[int((class_IDs[0][i])[0])]
                current_score = (scores[0][i])[0]
                current_bb = bounding_boxs[0][i-1]
                if current_class_id == 'traffic light':
                    traffic_boxes.append(current_bb)
        gc.collect()

        # print("Class ID: ", current_class_id)
        # print("Score: ", current_score)
        # print("Bounding Box Coordinates: ", current_bb, "\n")

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # vs.release()
        cv2.destroyAllWindows()

        return traffic_boxes, img

    # this function is now in Sensors.py
    # def get_gray_value(self, coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
    #     imgWidth = img.shape[1] # Get width
    #     x = round(coordinates[0]) # x translates directly
    #     y = imgWidth - round(coordinates[1]) # y is inverted
    #     self.cur_gps = (x,y)
    #     gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
    #     self.cur_gray_val = gray_val
    #     return gray_val

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
        current_gray_val = car.cc.sensor.get_gray_value(coordinates, self.regions_img)
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

    def update_desired_gray_val(self, region):
        if region == gp.region_dict['Region 1'] or region == gp.region_dict['Region 4']:
            self.predict.set_gray_val(215)
        else:
            self.predict.set_gray_val(215)



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
    else:
        car.update_desired_gray_val(cur_region)

    desired_coordinates, des_x, des_y, desired_region = car.get_next_coordinates()
    print("Initial waypoint coordinates: ", des_x, des_y)

    # Begin Driving
    # YOLO test car.cc.drive(0.6)  # get the car moving
    time.sleep(0.1)  # ...but only briefly
    # YOLO test car.cc.drive(car.speed)  # get the car moving again
    restart_car = False
    stop_at_light = False

    try:
        # Start driving!
        while True:

            ##### Milestone 3 - Check for objects first! #####
            car.cc.update_sensors()
            object_detected, image = car.cc.detector.detect_object() # Search region in front of car for object
#            cv2.imshow('vid', image)
#            cv2.waitKey(25)

            #### Implement YOLOv3MXNet ####
            coordinates = car.cc.sensor.get_gps_coord("Blue")
            # if the car is in the yolo region and it hasn't detected a green light yet, run yolo
            if car.cc.sensor.get_gray_value(coordinates, car.yolo_map)[0] == car.yolo_region_color :
                #yo = Yolo()
                car.yolo_region = True
                if car.green_light == False :
                    car.yolo_frame_count += 1 # = 10

                    if car.yolo_frame_count == 10 : # not sure how many frames we should count before we check YOLO. # monte carlo
                        car.yolo_frame_count = 0
                        print("DEBUG: YOLO")
                        #cv2.imshow("light", frame)
                        #cv2.waitKey(0)
                        print("going into Yolo")
                        bounding_boxes, yolo_img = car.main_yolo(frame)
                        light_boxes = []
                        # bounding_box = [x1, y1, x2, y2]   # format of bounding_boxes[i]
                        for box in range(0, len(bounding_boxes)):
                            if bounding_boxes[box][0] > img_middle and bounding_boxes[box][2] > img_middle :  # bounding box is on the right side of the camera
                                light_boxes.append(bounding_boxes[box])
                                print (light_boxes[-1])
                        y_of_light = 400 # arbitrary value that is used to compare when there is more than one detected traffic light
                        if not light_boxes:
                            print("DEBUG: oh no! there aren't any boxes!") # exit frame and try again
                        # we only want to look at one light, so if we detect more than one,
                        # we will look at the traffic light that is closest to the top of the pic as
                        # that one is likely to be the one we want to look at
                        else:
                            if len(light_boxes) > 1 :
                                for i in range(0, len(light_boxes)) :
                                    top_y = min(light_boxes[i][1], light_boxes[i][3])
                                    if top_y < y_of_light :
                                        y_of_light = top_y
                                        desired_light = i
                            else :
                                desired_light = 0   # there's only one traffic light detected in the desired region

                            # crop image:
                            x1 = int(light_boxes[desired_light][0])
                            y1 = int(light_boxes[desired_light][1])
                            x2 = int(light_boxes[desired_light][2])
                            y2 = int(light_boxes[desired_light][3])
                            cropped_img = yolo_img[y1:y2, x1:x2]

                            car.color_detected = car.predict_color(cropped_img)
                            # print(car.color_detected, " is the winner!")
                            # cv2.imshow("cropped", cropped_img)
                            # cv2.waitKey(0)
                            ################## I need to double check that y = 0 is the top ############
            else:
                car.yolo_region = False    # we are not currently in the region to check for traffic lights
                car.color_detected = 'black'   # make sure it's not an actual color we are detecting
                car.green_light = False # set this back to false so we don't lock out the function when we're in the region again
            # end of YOLO


            # YOLO
            if car.yolo_region:
                print("in yolo region")
                print("The light is: ", car.color_detected)
                if car.color_detected == 'green':
                    print("GO! it's a green light")
                    car.green_light = True
                    if stop_at_light:
                        restart_car = True
                        stop_at_light = False
                else :  # red or yellow light has been detected
                    car.green_light = False
                    # slow down car
                    print("stopping: red or yellow light")
                    car.cc.drive(0.0)
                    print('Its not green!')
                    stop_at_light = True
                    continue    # Skip all the remaining steps until the object is gone
            else : # debugging purposes
                print("NOT in yolo region")
            # end YOLO

            if (object_detected):
                car.cc.drive(0.0)
                print('Object Detected!')
                time.sleep(0.5)
                restart_car = True # When the object is removed, this tells the car to start again
                continue           # Skip all the remaining steps until the object is gone

            # YOLO test
            #if (restart_car):
            #    car.cc.drive(0.9)#0.6)  # get the car moving
            #    time.sleep(0.1)  # ...but only briefly
            #    car.cc.drive(car.speed)  # get the car moving again
            #    restart_car = False

            # object_detected = False

            ##################################################


            ''' YOLO
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
            '''
        car.cc.drive(0)  # stop the car
        print("Terminating Program")
        car.out_file.close()  # close the log file

    except KeyboardInterrupt:  # if the user Ctrl+C's us
        print("User Terminated Program")
        car.out_file.close()  # close the log file
