# Lane Testing

from LaneFollower import LaneFollower
from CarControl import CarControl
from matplotlib import pyplot as plt
import argparse
import imutils
import numpy as np
import cv2

leftColorMin = [90, 245, 180]        # Yellow - Determined by plotting imgHSV and hovering over the colors
leftColorMax = [100, 255, 210]       # Yellow
rightColorMin = [70, 60, 195]         # White
rightColorMax = [85, 80, 210]        # White

LF = LaneFollower()

vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
# M = np.load("M.npy")

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break


    LF.update_picture(frame)

    img = LF.convert_to_HSV(frame)
    leftMask = LF.filter_by_color(img, leftColorMin, leftColorMax)
    rightMask = LF.filter_by_color(img, rightColorMin, rightColorMax)

    leftCanny = LF.canny_img(leftMask)
    rightCanny = LF.canny_img(rightMask)

    leftCropped = LF.crop_image(leftCanny)
    rightCropped = LF.crop_image(rightCanny)

    leftLines = LF.hough_lines(leftCropped)
    rightLines = LF.hough_lines(rightCropped)


    try:
        left_line_x, left_line_y, right_line_x, right_line_y = LF.find_lines(leftLines, rightLines)

        leftFinal, left_points = LF.calculate_lines(rawImg, left_line_x, left_line_y, 1)
        rightFinal, right_points = LF.calculate_lines(leftFinal, right_line_x, right_line_y, 1)

        image = rightFinal
        print("Success!")
    except:
        image = (leftCropped | rightCropped)


    cv2.imshow("Camera Feed", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
vs.release()
cv2.destroyAllWindows()



