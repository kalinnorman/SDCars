""" Object Detection Test 
Used to test object detection functionality.
Pulls in a depth image, converts it to colors, then transforms to birdseye.
This is then cannied and cropped, and a specific region is searched for objects

Date: 2 Nov 2019
Author: Benj
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

referenceImage_path = ""
# saveName = "depthImage.jpg" 
min_compare_value = 3
reference_image = cv2.imread(referenceImage_path,0)

y_min = 118
y_max = 163 #193
x_min = 81
x_max = 107

# crop_y_min = 0
# crop_y_max = 163
v_tl = (25, crop_y_min)
v_bl = (75, crop_y_max)
v_br = (120, crop_y_max)
v_tr = (160, crop_y_min)
# y_crop_offset = (200 - crop_y_max)

count = 0

def crop_image(img):
        """
        Tkes in an image and crops out a specified range
        """
        cropVertices = [v_tl, v_bl, v_br, v_tr]                      # Corners of cropped image

        # Blank matrix that matches the image height/width
        mask = np.zeros_like(img)

        match_mask_color = 255 # Set to 255 to account for grayscale

        cv2.fillPoly(mask, np.array([cropVertices], np.int32), match_mask_color) # Fill polygon

        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

def detect_object(img):
    """
    Takes in image and searches each pixel in a designated region to see if there is an object
    """
    height = img.shape[0]
    width = img.shape[1]

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if img[y][x] > min_compare_value:
                return True

    return False

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
birdseye_transform_matrix = np.load('car_perspective_transform_matrix_warp_2.npy')

# Start streaming
pipeline.start(config)
reference_image = cv2.imread(referenceImage_path)
count = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        birdseye_frame = cv2.warpPerspective(gray_image, birdseye_transform_matrix, (200,200))
        bCanny = cv2.Canny(birdseye_frame, 100, 200)
        cropped_image  = crop_image(bCanny)
        threshold_image = cv2.subtract(reference_image, cropped_image)

        objectFound = False
        objectFound = detect_object(threshold_image)

        if objectFound:
            count += 1
            print("Object found! ", count)

finally:

    # Stop streaming
    pipeline.stop()


# Code below can be added above to plot the rectangular search region

# img = bCanny
# cv2.line(img, v_tl, v_bl, (255,0,0), 2)
# cv2.line(img, v_bl, v_br, (255,0,0), 2)
# cv2.line(img, v_br, v_tr, (255,0,0), 2)
# cv2.line(img, v_tr, v_tl, (255,0,0), 2)

# cv2.line(img, (x_min, y_min-y_crop_offset), (x_min, y_max-y_crop_offset), (255,0,0), 1)
# cv2.line(img, (x_min, y_max-y_crop_offset), (x_max, y_max-y_crop_offset), (255,0,0), 1)
# cv2.line(img, (x_max, y_max-y_crop_offset), (x_max, y_min-y_crop_offset), (255,0,0), 1)
# cv2.line(img, (x_max, y_min - y_crop_offset), (x_min, y_min - y_crop_offset), (255,0, 0), 1)

# Code below can be added to conver color to birdseye

# color_image = np.asanyarray(color_frame.get_data())   
# color_frame = cv2.applyColorMap(cv2.convertScaleAbs(color_frame, alpha=0.03), cv2.COLORMAP_JET)
# dCanny = cv2.Canny(depth_colormap, 50, 200)
# color_be_frame = cv2.warpPerspective(color_image, birdseye_transform_matrix, (200,200))




