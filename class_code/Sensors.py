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

"""
# YOLO USAGE
# sudo MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 yolo.py
# OPTIONAL PARAMETERS
# -c/--confidence (.0-1.0) (detected objects with a confidence higher than this will be used)
"""

import pyrealsense2 as rs
import time
import cv2
import numpy as np
import gc
import requests # needed to for get_gps_coord()
from Yolo import Yolo
# YOLO packages
from matplotlib import pyplot as plt
from gluoncv import model_zoo, utils
import pyrealsense2 as rs
from PIL import Image
from signal import signal, SIGINT
from sys import exit
import numpy as np
import mxnet as mx
import argparse
import imutils
import serial
import time
import cv2
import os
import gc
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

"""Transforms for YOLO series."""
def transform_test(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
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


def load_test(filenames, short=416):
    if not isinstance(filenames, list):
        filenames = [filenames]
    imgs = [letterbox_image(f, short) for f in filenames]
    return transform_test(imgs)


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size=416):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size

    scale = min(size / iw, size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (size, size), (128, 128, 128))
    new_image.paste(image, ((size - nw) // 2, (size - nh) // 2))
    return mx.nd.array(np.array(new_image))


# initialize the video stream, pointer to output video file, and
# frame dimensions
# vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)  # ls -ltr /dev/video*
# (W, H) = (None, None)

# Implement YOLOv3MXNet
net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

# Set device to GPU
device = mx.gpu()

net.collect_params().reset_ctx(device)

print('Running. Press CTRL-C to exit')


class Sensors():

    def __init__(self):
        # initialize variables that will contain the data
        self.current_depth = None
        self.current_rgb = None
        self.current_gyro = None
        self.current_accel = None
        self.current_class_id = None
        self.current_score = None
        self.current_bb = [0, 0, 0, 0]
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
        self.img_middle = 208    # this is the middle of the yolo picture, the width is always 416 pixels
        self.green_light = False
        self.traffic_boxes = []
        self.color_detected = "purple"
        signal(SIGINT, self.handler)


    # YOLO
    # Function to correctly exit program
    def handler(self, signal_received, frame):
        self.vs.release()
        cv2.destroyAllWindows()
        print('CTRL-C detected. Exiting gracefully')
        exit(0)

    # YOLO
    def find_color(self, img, color):
        # Convert image to HSV
        imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        print("Shape of imghsv", imghsv.shape)
        cv2.imwrite('cropped_image.jpg', imghsv)


        # Define the desired colorspace
        if color == 'red':
            lower = np.array([100, 40, 40], dtype='uint8') # was [150, 40, 40]
            upper = np.array([255, 255, 255], dtype='uint8')
        elif color == 'green':
            lower = np.array([50, 40, 40], dtype='uint8') 
            upper = np.array([100, 255, 255], dtype='uint8')
        elif color == 'yellow':
            lower = np.array([0, 40, 40], dtype='uint8') #np.array([0, 40, 40], dtype='uint8')
            upper = np.array([50, 255, 255], dtype='uint8')
        else:
            print("Choose a valid color, bro.")

        # Threshold the HSV image to get only the desired color
        mask = cv2.inRange(imghsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)
        count = cv2.countNonZero(res[:,:,0])

        return res, count  # returns the image and the count of non-zero pixels

    # YOLO
    def predict_color(self, img):

        colors = ['red', 'yellow', 'green']
        counts = []

        for color in colors:
            res, count = self.find_color(img, color)
            counts.append(count)
        print(counts)

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

    def update_sensors(self, yolo_flag=False):
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
        coordinates = self.get_gps_coord("Blue")

        #### Implement YOLOv3MXNet ####
        if yolo_flag:
            # from gluoncv import data
            yolo_image = Image.fromarray(frame, 'RGB')
            x, img = load_test(yolo_image, short=416)

            class_IDs, scores, bounding_boxs = net(x.copyto(device))

            # The next two lines draw boxes around detected objects
            ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
            #plt.show()

            # print(class_IDs)
            # print(scores)
            # print(bounding_boxs)

            # Convert to numpy arrays, then to lists
            class_IDs = class_IDs.asnumpy().tolist()
            scores = scores.asnumpy().tolist()
            bounding_boxs = bounding_boxs.asnumpy()
            self.traffic_boxes = []

            # iterate through detected objects
            for i in range(len(class_IDs[0])):
                if ((scores[0][i])[0]) > args["confidence"]:
                    current_class_id = net.classes[int((class_IDs[0][i])[0])]
                    current_score = (scores[0][i])[0]
                    self.current_bb = bounding_boxs[0][i - 1]
                    if current_class_id == 'traffic light':
                        self.traffic_boxes.append(self.current_bb)

            if len(class_IDs[0]) == 0:
                self.current_bb = [0, 0, 0, 0]
                print("YOLO didn't find anything. :(")

            #print("Bounding Box Coordinates: ", self.current_bb)
            #cv2.imshow("Camera Feed", frame)

            light_boxes = []
            # bounding_box = [x1, y1, x2, y2]   # format of bounding_boxes[i]
            for box in range(0, len(self.traffic_boxes)):
                if self.traffic_boxes[box][0] > self.img_middle and self.traffic_boxes[box][2] > self.img_middle:  # bounding box is on the right side of the camera
                    light_boxes.append(self.traffic_boxes[box])
                    print(light_boxes[-1])
            y_of_light = 400  # arbitrary value that is used to compare when there is more than one detected traffic light
            if not light_boxes:
                print("DEBUG: oh no! there aren't any boxes!")  # exit frame and try again
                ########## we don't care about the rest of the code
            # we only want to look at one light, so if we detect more than one,
            # we will look at the traffic light that is closest to the top of the pic as
            # that one is likely to be the one we want to look at
            else:
                print("light boxes = ", len(light_boxes))
                if len(light_boxes) > 1:
                    for i in range(0, len(light_boxes)):
                        top_y = min(light_boxes[i][1], light_boxes[i][3])
                        if top_y < y_of_light:
                            y_of_light = top_y
                            desired_light = i
                else:
                    desired_light = 0  # there's only one traffic light detected in the desired region

                # crop image:
                x1 = int(light_boxes[desired_light][0])
                y1 = int(light_boxes[desired_light][1])
                x2 = int(light_boxes[desired_light][2])
                y2 = int(light_boxes[desired_light][3])
                cropped_img = img[y1:y2, x1:x2]
                self.color_detected = self.predict_color(cropped_img)
                print(self.color_detected, " is the winner!")
                ################## I need to double check that y = 0 is the top ############
        #### END OF YOLO ####

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
