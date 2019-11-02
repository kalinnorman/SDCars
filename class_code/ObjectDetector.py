import pyrealsense2 as rs 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 

class ObjectDetector:
    """
    This class is used to detect objects
    """

    def __init__(self, sensor):

        self.sensor = sensor
        self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix_warp_2.npy')

    def crop_image(self, img):
        """
        Takes in an image and crops out a specified range.
        In this case, we are cropping a trapezoid out of the birdseye view
        """
        cropVertices = [(25, 0),                      # Corners of cropped image
                        (75, 163),        # Gets bottom portion
                        (120, 163),
                        (160, 0)] 

        # Blank matrix that matches the image height/width
        mask = np.zeros_like(img)

        match_mask_color = 255 # Set to 255 to account for grayscale

        cv2.fillPoly(mask, np.array([cropVertices], np.int32), match_mask_color) # Fill polygon

        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def search_range(self, img):
        """
        Searches through a picture for non-zero values
        """
        print("starting search range...")
        height = img.shape[0]
        width = img.shape[1]

        for y in range(115, 164):
            for x in range(25, 160):
                if img[y][x] != 0:
                    return True

        return False

    def detect_object(self):
        """
        Main class method
        Loads in a depth image, converts to birdseye view, Cannies, and then crops.

        """
        try:
            print("Trying to detect...")
            time, depth_image = sensor.get_depth_data()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            print("Color map...")

            birdseye_frame = cv2.warpPerspective(depth_colormap, self.birdseye_transform_matrix, (200, 200))

            print("warped...")
            cannied_image = cv2.Canny(birdseye_frame, 50, 200) 

            print("cannied...")

            cropped_image = self.crop_image(cannied_image)

            object_found = self.search_range(cropped_image)
            print("Returning ", object_found)
            return object_found
        except:
            print("Detect Image Failed")
            return False
