
"""
lane_detection_test_script.py
Author: redd
"""

import cv2
import numpy as np
from road_detection import RoadDetection
#from LaneFollower import LaneFollower
from ReddFollower import ReddFollower


birdseye_transform_matrix = np.load('car_perspective_transform_matrix.npy')

def modify_video(filename='Oct4DriveTest2.avi'):
    cap = cv2.VideoCapture(filename)

    #rd = RoadDetection()
    #lf = LaneFollower()
    rf = ReddFollower()

    while cap.isOpened():
        ret, frame = cap.read()  # for video, image shape is (480, 640, 3)

        # birdseye_frame = cv2.warpPerspective(frame, birdseye_transform_matrix, (400, 400))

        # gray[0:img.shape[0], 0:img.shape[1]] = img
        #rd.find_straight_road_2(frame)
        # lf.runLaneDetection(frame)
        result, white, yellow, white_edges, yellow_edges = rf.find_lanes(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('filtered', result)
        cv2.imshow('misc', white_edges)
        cv2.imshow('yellow', yellow_edges)

        key_press = cv2.waitKey(25)

        if key_press & 0xFF == ord('q'):
            break
        if key_press & 0xFF == ord('p'): # take a picture
            cv2.imwrite('saved_frame.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()

modify_video(filename='Oct4DriveTest2.avi')
