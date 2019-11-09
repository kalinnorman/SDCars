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

class ObjectDetector:
    """
    This class is used to detect objects
    """

    def __init__(self, sensor):

        self.sensor = sensor       # Passed in by CarControl.py
        self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix_warp_2.npy')
        self.count = 0
        self.debuggerCount = 3

    def crop_image(self, img):
        """
        Takes in an image and crops out a specified range.
        In this case, we are cropping a trapezoid out of the birdseye view
        """
        cropVertices = [(25, 0),          # Corners of trapezoid to crop
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

        for y in range(90, 180):       # 115 <= y <= 164
            for x in range(75, 110):    # 25 <= x <= 160
                if img[y][x] != 0:
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

            birdseye_frame = cv2.warpPerspective(depth_colormap, self.birdseye_transform_matrix, (200, 200))
            cannied_image = cv2.Canny(birdseye_frame, 50, 200) 

            cropped_image = self.crop_image(cannied_image)

            object_found = self.search_range(cropped_image)
            
            if (object_found):
                self.count = self.count + 1
                if (self.count >= self.debuggerCount):
                   # self.count = 0
                    return True, cropped_image
            else:
                self.count = 0
            return False, cropped_image
        except:
            print("Detect Image Failed")
            return False, 0
