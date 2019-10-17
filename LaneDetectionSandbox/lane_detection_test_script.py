
"""
lane_detection_test_script.py
Author: redd
"""

import cv2
import numpy as np
from road_detection import RoadDetection
#from LaneFollower import LaneFollower
from ReddFollower import ReddFollower


def modify_video(filename='Oct4DriveTest2.avi'):
    cap = cv2.VideoCapture(filename)

    #rd = RoadDetection()
    #lf = LaneFollower()
    rf = ReddFollower()

    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray[0:img.shape[0], 0:img.shape[1]] = img
        #rd.find_straight_road_2(frame)
        # lf.runLaneDetection(frame)
        result, misc = rf.find_lanes(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('filtered', result)
        cv2.imshow('misc', misc)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

modify_video()
