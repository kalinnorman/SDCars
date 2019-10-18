# Lane Testing using birds eye view

from LaneFollower import LaneFollower
# from CarControl import CarControl
from matplotlib import pyplot as plt
import argparse
import imutils
import time
import numpy as np
import cv2

LF = LaneFollower()
count = 0
# vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
vs = cv2.VideoCapture("C:/Users/benjj/Documents/College/Fall2019/ECEN522/Code/SDCars/class_code/TestVideo.avi")

for lamePhoto in range(1,10):
    (grabbed, frame) = vs.read()

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    frame = LF.get_birds_eye_view(frame)
    LF.update_picture(frame)
    img = LF.convert_to_HSV(frame)

    LF.update_picture(img)

    leftMask, rightMask = LF.filter_by_color(img)

    leftCanny = LF.canny_img(leftMask)
    rightCanny = LF.canny_img(rightMask)

    leftCropped = LF.crop_image(leftCanny)
    rightCropped = LF.crop_image(rightCanny)

    leftLines = LF.hough_lines(leftCropped)
    rightLines = LF.hough_lines(rightCropped)

    try:
        left_line_x, left_line_y, right_line_x, right_line_y = LF.find_lines(leftLines, rightLines)
        leftFinal, left_points = LF.calculate_lines(frame, left_line_x, left_line_y, 1, 3)
        rightFinal, right_points = LF.calculate_lines(leftFinal, right_line_x, right_line_y, 1, 3)
        lane_final, mid_points = LF.calculate_center(rightFinal, left_points, right_points, 1)
        frame = lane_final

        ## NOT USABLE FOR BIRDSEYE VIEW
        # int_point = LF.find_intersection(left_points[0][0], left_points[0][1],
        #                         left_points[1][0], left_points[1][1],
        #                         right_points[0][0], right_points[0][1],
        #                         right_points[1][0], right_points[1][1])
        # frame = LF.plot_center(frame, int_point)

        angle = LF.calculate_angle(mid_points[5])

        print(angle)

    except:
        frame = frame

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    count = count + 1
    time.sleep(0.01)
       
    
vs.release()
cv2.destroyAllWindows()



