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


def crop_image(img):
        """
        Takes in an image and crops out a specified range
        """
        cropVertices = [(25, 0),                      # Corners of cropped image
                        (75, 163),        # Gets bottom portion
                        (120, 163),
                        (160, 0) ] 

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

    for y in range(124, 164):
        for x in range(25, 160):
            if img[y][x] != 0:
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
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        dCanny = cv2.Canny(depth_colormap, 50, 200)

        birdseye_frame = cv2.warpPerspective(depth_colormap, birdseye_transform_matrix, (200,200))

        bCanny = cv2.Canny(birdseye_frame, 50, 200)

        cropped_image  = crop_image(bCanny)



        # Show images
        # plt.imshow(cropped_image)
        # plt.show()
        objectFound = False
        objectFound = detect_object(cropped_image)

        if objectFound:
            print("Object found!")


finally:

    # Stop streaming
    pipeline.stop()



