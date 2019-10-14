
# USAGE: python3 take_picture.py

# import the necessary packages
import argparse
import imutils
import cv2

vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*

i=0
while i < 10:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    i+=1
    
cv2.imwrite("calibration_image.jpg", frame)

vs.release()
