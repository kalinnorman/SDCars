"""
Sensors.py
A class for handling sensor data
This includes camera, IMU, GPS, YOLO3...
Functions & parameters:
init()
get_all_data()
# Functions to access camera/IMU data
get_depth_data()
get_rgb_data()
get_gyro_data()
get_accel_data()
# Functions for GPS
get_region()
# Functions to access yolo data
get_class_id()
get_score()
get_bb()
# Functions for accessing IMU data
gyro_data(gyro)
accel_data(accel)
update_sensors()
# Functions for GPS
get_gps_region()
get_gps_coord(color)
Author: KAB'B
"""

import pyrealsense2 as rs
import time
import cv2
import numpy as np
import gc
import requests # needed to for get_gps_coord()


class Sensors():

    def __init__(self):
        # initialize variables that will contain the data
        self.current_depth = None
        self.current_rgb = None
        self.current_gyro = None
        self.current_accel = None
        self.current_class_id = None
        self.current_score = None
        self.current_bb = None
        self.region = None

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        self.vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)  # ls -ltr /dev/video*
        self.writer = None
        (self.W, self.H) = (None, None)

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        # Start streaming
        self.pipeline.start(self.config)

        # YOLO
        self.yolo_map = cv2.imread('Maps/yolo_regions.bmp')
        self.yolo_region_color = 123
        self.yolo_region = False
        self.img_middle = 208    # this is the middle of the yolo picture, the width is always 416 pixels
        self.yolo_frame_count = 0    # we use this so that we aren't checking yolo at every frame; probably should put this in Sensors.py
        self.yo = YOLO()
        self.green_light = False

    # YOLO
    # this is also in predictive_drive_car.py
    def get_gray_value(self, coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
        imgWidth = img.shape[1] # Get width
        x = round(coordinates[0]) # x translates directly
        y = imgWidth - round(coordinates[1]) # y is inverted
        cur_gps = (x,y)
        gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
        cur_gray_val = gray_val
        return gray_val

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

    # Data is from IMU, camera, and YOLO3
    def get_all_data(self):
        """
        Returns all data as a list
        """
        return (time.time(),
                self.current_depth,
                self.current_rgb,
                self.current_gyro,
                self.current_accel,
                self.current_class_id,
                self.current_score,
                self.current_bb)

    # Functions to access camera/IMU data
    def get_depth_data(self):
        return time.time(), self.current_depth

    def get_rgb_data(self): # gives a picture frame
        return time.time(), self.current_rgb

    def get_gyro_data(self):
        return time.time(), self.current_gyro

    def get_accel_data(self):
        return time.time(), self.current_accel

    # Functions for GPS
    def get_region(self):
        return time.time(), self.region

    # Functions to access yolo data
    def get_class_id(self):
        return time.time(), self.current_class_id

    def get_score(self):
        return time.time(), self.current_score

    def get_bb(self): # bounding box for YOLO3
        return time.time(), self.current_bb

    # Functions for accessing IMU data
    def gyro_data(self, gyro):
        return np.asarray([gyro.x, gyro.y, gyro.z])

    def accel_data(self, accel):
        return np.asarray([accel.x, accel.y, accel.z])

    def update_sensors(self):
        # read the next frame from the file
        (grabbed, frame) = self.vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            raise Exception("No frame grabbed")
            #break

        # if the frame dimensions are empty, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # start realsense pipeline
        rsframes = self.pipeline.wait_for_frames()

        # GPS data
        self.region = self.get_gps_region()

        #### Implement YOLOv3MXNet ####
        coordinates = self.get_gps_coord("Blue")
        # if the car is in the yolo region and it hasn't detected a green light yet, run yolo
        if self.get_gray_value(coordinates, self.yolo_map)[0] == self.yolo_region_color :
            self.yolo_region = True
            if self.green_light == False :
                self.yolo_frame_count += 1 # = 10

                if self.yolo_frame_count == 10 : # not sure how many frames we should count before we check YOLO. # monte carlo
                    self.yolo_frame_count = 0

                    cv2.imshow("light", frame)
                    cv2.waitKey(0)

                    bounding_boxes, yolo_img = self.yo.main_yolo(frame)#imgLight)#, args)
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
                        elif len(light_boxes) > 1 :
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

                        self.color_detected = self.predict_color(cropped_img)
                        # print(self.color_detected, " is the winner!")
                        # cv2.imshow("cropped", cropped_img)
                        # cv2.waitKey(0)
                        ################## I need to double check that y = 0 is the top ############
        else:
            self.yolo_region = False    # we are not currently in the region to check for traffic lights
            self.color_detected = 'black'   # make sure it's not an actual color we are detecting
            self.green_light = False # set this back to false so we don't lock out the function when we're in the region again
        # end of YOLO

        # iterate through camera/IMU data, updating global variable
        for rsframe in rsframes:
            # Retrieve IMU data
            if rsframe.is_motion_frame():
                self.current_accel = self.accel_data(rsframe.as_motion_frame().get_motion_data())
                self.current_gyro = self.gyro_data(rsframe.as_motion_frame().get_motion_data())
            # Retrieve depth data
            if rsframe.is_depth_frame():
                depth_frame = rsframes.get_depth_frame()
                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                self.current_depth = depth_image
                self.current_rgb = frame

        gc.collect()  # collect garbage

    # call this when car has reached an intersection
    def get_gps_region(self):
        x,y = self.get_gps_coord("Blue")  # outputs coordinates (x,y)

        # arbitrary values, need to test when have testing space
        # we only care about the horizontal axis (y) for turning purposes
        if y > 1400 :
            region = 'north'
            # should go left
        elif y > 1200 :
            region = 'middle north'
            # should go right
        elif y < 200 :
            region = 'south'
            # should go left
        elif y < 500 :
            region = 'middle south'
            # should go right
        else :
            region = 'middle'
            # can go any direction
        return region

    # retrieves the coordinates of a car (provided in class code)
    # car color options are: "Green", "Red", "Purple", "Blue", "Yellow"
    # my advice: pick "Blue" -ABT
    def get_gps_coord(self, color):
        # api-endpoint
        success = False
        while not success:
            URL = "http://192.168.1.8:8080/%s" % color

            # sending get request and saving the response as response object
            r = requests.get(url = URL)

            # extracting data
            coorString = r.text
            try:
                coordinates = coorString.split()
                latitude = float(coordinates[0])
                longitude = float(coordinates[1])
                success = True
            except:
                success = False

        return (latitude, longitude)
