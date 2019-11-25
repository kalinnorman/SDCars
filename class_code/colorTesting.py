import numpy as np 
from matplotlib import pyplot as plt 
import cv2 

import pyrealsense2 as rs 

colorMin_1 = np.asarray([230, 0, 0])
colorMax_1 = np.asarray([255, 255, 255])
colorMin_2 = np.asarray([0, 240, 0])
colorMax_2 = np.asarray([255, 255, 255])
colorMin_3 = np.asarray([0, 0, 0])
colorMax_3 = np.asarray([10, 10, 255])


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        for x in range(10):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

        color_image = np.asanyarray(color_frame.get_data())
#        color_frame = cv2.applyColorMap(cv2.convertScaleAbs(color_frame, alpha= 0.03), cv2.COLORMAP_JET)

#        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        img = color_image
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
        #cv2.imshow("image", imageStack)
        #cv2.waitKey(0)
finally:
    pipeline.stop()
