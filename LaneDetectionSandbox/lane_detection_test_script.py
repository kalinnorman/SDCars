
"""
lane_detection_test_script.py
Author: redd
"""

import cv2
import numpy as np
from road_detection import RoadDetection
#from LaneFollower import LaneFollower
from ReddFollower import ReddFollower


def modify_video(filename='Oct4DriveTest.avi'):
    cap = cv2.VideoCapture(filename)

    #rd = RoadDetection()
    #lf = LaneFollower()
    rf = ReddFollower()

    while cap.isOpened():
        ret, frame = cap.read()  # for video, image shape is (480, 640, 3)

        # gray[0:img.shape[0], 0:img.shape[1]] = img
        #rd.find_straight_road_2(frame)
        # lf.runLaneDetection(frame)
        result, control_values = rf.find_lanes(frame, show_images=False)


        print(control_values)

        key_press = cv2.waitKey(2)

        if key_press & 0xFF == ord('q'):
            break
        if key_press & 0xFF == ord('p'): # take a picture
            cv2.imwrite('saved_frame.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()

modify_video(filename='Oct4DriveTest2.avi')
