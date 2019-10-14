# Lane Testing

from LaneFollower import LaneFollower
from car_control import carControl
from matplotlib import pyplot as plt
import argparse
import imutils
import numpy as np
import cv2

LF = LaneFollower()

vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
M = np.load("M.npy")

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

    image = (leftCropped | rightCropped)


    cv2.imshow("Camera Feed", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
vs.release()
cv2.destroyAllWindoes()



