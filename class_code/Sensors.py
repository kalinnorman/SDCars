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
        #net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

        # from gluoncv import data
        #yolo_image = Image.fromarray(frame, 'RGB')
        #x, img = load_test(yolo_image, short=416)

        # Set device to GPU
        #device = mx.gpu()

        #net.collect_params().reset_ctx(device)

        #class_IDs, scores, bounding_boxs = net(x.copyto(device))

        # Convert to numpy arrays, then to lists
        #class_IDs = class_IDs.asnumpy().tolist()
        #scores = scores.asnumpy().tolist()
        #bounding_boxs = bounding_boxs.asnumpy()

        # iterate through detected objects, updating global variable
        # for i in range(len(class_IDs[0])):
        #     if ((scores[0][i])[0]) > args["confidence"]:
        #         current_class_id = net.classes[int((class_IDs[0][i])[0])]
        #         current_score = (scores[0][i])[0]
        #         current_bb = bounding_boxs[0][i]

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
        URL = "http://192.168.1.8:8080/%s" % color

        # sending get request and saving the response as response object
        r = requests.get(url = URL)

        # extracting data
        coorString = r.text
        coordinates = coorString.split()
        latitude = float(coordinates[0])
        longitude = float(coordinates[1])
        return (latitude, longitude)
