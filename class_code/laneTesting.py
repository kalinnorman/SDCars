# Lane Testing

from LaneFollower import LaneFollower
# from CarControl import CarControl
from matplotlib import pyplot as plt
import argparse
import imutils
import time
import numpy as np
import cv2

#    OG Values
#leftColorMin = [90, 245, 180]
#leftColorMax = [100, 255, 210]
#rightColorMin = [5, 15, 170]       # White   - Updated values
#rightColorMax = [20, 40, 230]        # White

# leftColorMin = [85, 240, 175]        # Yellow - Determined by plotting imgHSV and hovering over the colors
# leftColorMax = [105, 255, 220]       # Yellow
# rightColorMin = [1, 10, 160]
# rightColorMax = [30, 65, 240]

LF = LaneFollower()
count = 0
vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
vs = cv2.VideoCapture("C:/Users/benjj/Documents/College/Fall2019/ECEN522/Code/SDCars/class_code/TestVideo.avi")

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    LF.update_picture(frame)

    img = LF.convert_to_HSV(frame)

    leftMask = LF.filter_by_color(img, True)
    rightMask = LF.filter_by_color(img, False)


    leftCanny = LF.canny_img(leftMask)
    rightCanny = LF.canny_img(rightMask)

    leftCropped = LF.crop_image(leftCanny)
    rightCropped = LF.crop_image(rightCanny)

    leftLines = LF.hough_lines(leftCropped)
    rightLines = LF.hough_lines(rightCropped)

    try:
        left_line_x, left_line_y, right_line_x, right_line_y = LF.find_lines(leftLines, rightLines)
        leftFinal, left_points = LF.calculate_lines(frame, left_line_x, left_line_y, 1)
        rightFinal, right_points = LF.calculate_lines(leftFinal, right_line_x, right_line_y, 1)
        frame = rightFinal

        int_point = LF.find_intersection(left_points[0][0], left_points[0][1],
                                left_points[1][0], left_points[1][1],
                                right_points[0][0], right_points[0][1],
                                right_points[1][0], right_points[1][1])
        frame = LF.plot_center(frame, int_point)

        angle = LF.calculate_angle(int_point)

        print(angle)

    except:
        frame = frame
#        try:
#            for x1, y1, x2, y2 in leftLines[0]:
#                cv2.line(frame, (x1,y1),(x2,y2),(0,255,255),2) 
#        except:
#            count = 1

    # if count >= 239:

    #     plt.figure(figsize=(20,8))
    #     plt.subplot(1,3,1)
    #     plt.imshow(frame)

    #     plt.subplot(1,3,2)
    #     plt.imshow(rightMask | leftMask)

    #     plt.subplot(1,3,3)
    #     plt.imshow(rightCropped | leftCropped)

    #     plt.title(count)
    #     plt.show()

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    count = count + 1
    time.sleep(0.1)
       
    
vs.release()
cv2.destroyAllWindows()



