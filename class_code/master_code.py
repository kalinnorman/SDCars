# master_code.py
'''
Use the following inputs to control the car
*************** Inputs ***************
From the Realsense camera:
    Depth Data
    RGB Data
    Gyroscope Data
    Accelerometer Data
From GPS system
    GPS Coordinates
From yolo
    Class ID
    Confidence
    Bounding Box Coordinates
**************************************
******* Controlling The Car **********
    steer(int degree)	1000 = Full left turn
    2000 = Full right turn
    1500 = (nearly) Straight
    drive(int speed)  	1000 = Fast reverse
    2000 = Fast forward
    1500 = (nearly) Stopped
    EXTREMELY IMPORTANT: Be very careful whe controlling the car.
    NEVER tell it to go full speed. Safely test the car to find
    a safe range for your particular car, and don't go beyond that
    speed. These cars can go very fast, and there is expensive hardware
    on them, so don't risk losing control of the car and breaking anything.
**************************************
'''

# USAGE
# sudo MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 master_code.py
# OPTIONAL PARAMETERS
# -c/--confidence (.0-1.0) (detected objects with a confidence higher than this will be used)

# import the necessary packages
from car_control import steer, drive
from matplotlib import pyplot as plt
from gluoncv import model_zoo, utils
import pyrealsense2 as rs
from PIL import Image
import numpy as np
import mxnet as mx
import argparse
import imutils
import serial
import time
import cv2
import os
import gc

import demo_steering as demo
from helpers import initialize_car

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", help="path to output video")
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

    scale = min(size/iw, size/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (size, size), (128, 128, 128))
    new_image.paste(image, ((size-nw)//2, (size-nh)//2))
    return mx.nd.array(np.array(new_image))

# Functions for accessing IMU data
def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

# initialize global variables that will contain the data
current_depth = None
current_rgb = None
current_gyro = None
current_accel = None
current_class_id = None
current_score = None
current_bb = None

def get_all_data():
   global current_depth
   global current_rgb
   global current_gyro
   global current_accel
   global current_class_id
   global current_score
   global current_bb
   return (time.time(), current_depth, current_rgb, current_gyro,
       current_accel, current_class_id, current_score, current_bb)

# Functions to access camera/IMU data
def get_depth_data():
    global current_depth
    return (time.time(), current_depth)

def get_rgb_data():
    global current_rgb
    return (time.time(), current_rgb)

def get_gyro_data():
    global current_gyro
    return (time.time(), current_gyro)

def get_accel_data():
    global current_accel
    return (time.time(), current_accel)

# Functions to access yolo data
def get_class_id():
    global current_class_id
    return (time.time(), current_class_id)

def get_score():
    global current_score
    return (time.time(), current_score)

def get_bb():
    global current_bb
    return (time.time(), current_bb)

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
writer = None
(W, H) = (None, None)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# Start streaming
pipeline.start(config)

# initialize communication with the arduino
initialize_car()


# loop over frames from the video file stream
try:
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # start realsense pipeline
        rsframes = pipeline.wait_for_frames()

        # Implement YOLOv3MXNet
        net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

        # from gluoncv import data
        yolo_image = Image.fromarray(frame, 'RGB')
        x, img = load_test(yolo_image, short=416)

        # Set device to GPU
        device=mx.gpu()

        net.collect_params().reset_ctx(device)

        class_IDs, scores, bounding_boxs = net(x.copyto(device))

        # Convert to numpy arrays, then to lists
        class_IDs = class_IDs.asnumpy().tolist()
        scores = scores.asnumpy().tolist()
        bounding_boxs = bounding_boxs.asnumpy()

        # iterate through detected objects, updating global variable
        for i in range(len(class_IDs[0])):
            if ((scores[0][i])[0]) > args["confidence"]:
                current_class_id = net.classes[int((class_IDs[0][i])[0])]
                current_score = (scores[0][i])[0]
                current_bb = bounding_boxs[0][i]

        # iterate through camera/IMU data, updating global variable
        for rsframe in rsframes:
            # Retrieve IMU data
            if rsframe.is_motion_frame():
                current_accel = accel_data(rsframe.as_motion_frame().get_motion_data())
                current_gyro = gyro_data(rsframe.as_motion_frame().get_motion_data())
            # Retrieve depth data
            if rsframe.is_depth_frame():
                depth_frame = rsframes.get_depth_frame()
                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                current_depth = depth_image
                current_rgb = frame

        gc.collect()

        '''
        Use the following functions to access the data.
        get_all_data returns the current time followed by all the data.
        The other functions return the current time and only the corresponding data.
        Store and use the data however you decide
        '''
        all_data = get_all_data()
        depth = get_depth_data()
        rgb = get_rgb_data()
        accel = get_accel_data()
        gyro = get_gyro_data()
        object_id = get_class_id()
        object_score = get_score()
        object_bb = get_bb() # bounding box coordinates (x, y, w, h). (x, y) are the

        print("Depth is of type ", type(depth),  " and contains: ", depth)
        print("RGB is of type ", type(rgb),  " and contains: ", rgb)
        print("Accel is of type ", type(accel),  " and contains: ", accel)
        print("Gyro is of type ", type(gyro),  " and contains: ", gyro)
        #print("ID is of type ", type(object_id),  " and contains: ", object_id)
        #print("Score is of type ", type(object_score),  " and contains: ", object_score)
        #print("Bounding Box is of type ", type(object_bb),  " and contains: ", object_bb)

        '''
        Controlling the Car
        Use the following functions to control the car:
        steer(int degree) - 1000 = Full left turn	2000 = Full right turn	1500 = (nearly) Straight
        drive(int speed) - 1000 = Fast reverse 	2000 = Fast forward	1500 = (nearly) Stopped
        	IMPORTANT: Never go full speed. See note near top of file.
            time.sleep(x) can be used in between function calls if needed, where x is time in seconds
        '''

        demo.demo_steering()

        '''
        # Example car control
        print("Turn right")
        steer(1800)
        time.sleep(1)
        print("Turn left")
        steer(1200)
        time.sleep(1)
        print("Turn straight")
        steer(1500)
        time.sleep(1)

        print("Drive Forward")
        drive(1700)
        time.sleep(1)
        print("Stop")
        drive(1500)
        time.sleep(1)
        print("Drive Backward")
        drive(1300)
        time.sleep(1)
        print("Stop")
        drive(1500)
        time.sleep(1)

        '''
except KeyboardInterrupt:

    drive(1500)
    steer(1500)

    #writer.release()
    vs.release()
