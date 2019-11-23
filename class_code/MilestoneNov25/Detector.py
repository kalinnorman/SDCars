""" Object Detector
Class that is used to detect objects in the close field of vision.
Gets an image, transforms it to birdseye view, cannies it, and then crops it.
A specified portion of the cropped image is searched for non-zero values (meaning an object)

Date: 2 Nov 2019
Author: Benj
"""

import pyrealsense2 as rs 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import time

class Detector:
    """
    This class is used to detect objects
    """

    def __init__(self, sensor):

        self.sensor = sensor       # Passed in by CarControl.py
        self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix_warp_2.npy')
        self.count = 0
        self.debuggerCount = 3 # Accounts for noise
        self.min_difference = 3 # Accounts for normal fluctuation when comparing to reference image
        self.reference_image = cv2.imread("/home/nvidia/Desktop/class_code/MilestoneNov25/referenceImage.jpg",0)

        # Search Region Parameters:
        self.y_min = 118
        self.y_max = 163 # (193?)
        self.x_min = 81
        self.x_max = 107

    def crop_image(self, img):
        """
        Takes in an image and crops out a specified range.
        In this case, we are cropping a trapezoid out of the birdseye view
        """
        cropVertices = [(25, 0),           # Corners of trapezoid to crop
                        (75, 163),    
                        (120, 163),
                        (160, 0)] 

    
        mask = np.zeros_like(img) # Blank matrix that matches the image height/width
        match_mask_color = 255    # Set to 255 to account for grayscale
        cv2.fillPoly(mask, np.array([cropVertices], np.int32), match_mask_color) # Fill polygon

        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def search_range(self, img):
        """
        Searches through a picture for non-zero values, returning True if something is found
        (Search area is in need of further tuning)
        """

        for y in range(self.y_min, self.y_max):# 115 <= y <= 164
            for x in range(self.x_min, self.x_max):    # 25 <= x <= 160
                if img[y][x] > self.min_difference:
                    return True

        return False

    def detect_object(self):
        """
        Main class method
        Loads in a depth image, converts to birdseye view, Cannies, and then crops.
        """
        try:
            time, depth_image = self.sensor.get_depth_data()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            grayed_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            birdseye_frame = cv2.warpPerspective(grayed_depth, self.birdseye_transform_matrix, (200, 200))
            # cannied_image = cv2.Canny(birdseye_frame, 50, 200) 
            cropped_image = self.crop_image(birdseye_frame)
            threshold_image = cv2.subtract(self.reference_image, cropped_image)

            object_found = self.search_range(threshold_image)
            
            if (object_found):
                self.count = self.count + 1
                if (self.count >= self.debuggerCount):
                   # self.count = 0 <- removed so it doesn't keep restarting and stopping
                    return True, threshold_image
            else:
                self.count = 0
            return False, threshold_image
        except:
            print("Detect Image Failed")
            return False, 0
