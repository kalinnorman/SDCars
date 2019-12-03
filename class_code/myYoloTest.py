import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
from YoloTest import Yolo
import pyrealsense2 as rs

# Setup stuff
yo = Yolo()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img", img)
        cv2.waitKey(0)

        bounding_boxes, yolo_img = yo.main_yolo(img)
        light_boxes = []
        # bounding_box = [x1, y1, x2, y2]   # format of bounding_boxes[i]
        for box in range(0, len(bounding_boxes)):
            if bounding_boxes[box][0] > img_middle and bounding_boxes[box][2] > img_middle :  # bounding box is on the right side of the camera
                light_boxes.append(bounding_boxes[box])
                print (light_boxes[-1])
        y_of_light = 400 # arbitrary value that is used to compare when there is more than one detected traffic light
        if not light_boxes:
            print("DEBUG: oh no! there aren't any boxes!") # exit frame and try again
        else :
            desired_light = 0   



finally:
    pipeline.stop()