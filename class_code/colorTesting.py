import numpy as np 
from matplotlib import pyplot as plt 
import cv2 

import pyrealsense2 as rs 

colorMin_1 = np.asarray([230, 0, 0])
colorMax_1 = np.asarray([255, 255, 255])
colorMin_2 = np.asarray([0, 220, 0])
colorMax_2 = np.asarray([100, 255, 100])
colorMin_3 = np.asarray([0, 0, 230])
colorMax_3 = np.asarray([60, 60, 255])


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

def crop_image(img):
        """
        Tkes in an image and crops out a specified range
        """
        cropVertices = [(350,0), (350,100), (450, 100), (450,0)]                      # Corners of cropped image

        # Blank matrix that matches the image height/width
        mask = np.zeros_like(img)

        match_mask_color = 255 # Set to 255 to account for grayscale

        cv2.fillPoly(mask, np.array([cropVertices], np.int32), match_mask_color) # Fill polygon

        masked_image = cv2.bitwise_and(img, mask)

        return masked_image


try:
    while True:
        for x in range(10):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

        color_image = np.asanyarray(color_frame.get_data())
#        color_frame = cv2.applyColorMap(cv2.convertScaleAbs(color_frame, alpha= 0.03), cv2.COLORMAP_JET)

        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
#        img = crop_image(color_image)
#        img = color_image
        mask_image_1 = cv2.inRange(img, colorMin_1, colorMax_1)
        mask_image_2 = cv2.inRange(img, colorMin_2, colorMax_2)
        mask_image_3 = cv2.inRange(img, colorMin_3, colorMax_3)
 
        #cv2.imshow("vid",img)
        #cv2.waitKey(0)
       
        plt.imshow(img)
        plt.show()
        imageStack = np.hstack((mask_image_1, mask_image_2, mask_image_3))
       
        plt.imshow(imageStack)
        plt.show()

finally:
    pipeline.stop()
