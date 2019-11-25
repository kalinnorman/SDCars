import numpy as np 
from matplotlib import pyplot as plt 
import cv2 

import pyrealsense2 as rs 

colorMin = np.asarray([85, 240, 175])
colorMax = np.asarray([105, 255, 220])


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_frame = cv2.applyColorMap(cv2.convertScaleAbs(color_frame, alpha= 0.03), cv2.COLORMAP_JET)

        img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        mask_image_1 = cv2.inRange(img, colorMin, colorMax)
        mask_image_2 = cv2.inRange(img, colorMin, colorMax)

        imageStack = np.hstack((mask_image_1, mask_image_2))

        cv2.imshow("image", imageStack)
        key=cv2.waitKey(0)
