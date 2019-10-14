#live_camera.py

# import the necessary packages
import argparse
import imutils
import numpy as np
import cv2
	

vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
M = np.load("M.npy")

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    height,width,_=frame.shape
    frame = cv2.warpPerspective(frame,M,(width,height))

    # Convert image to gray scale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert to HSV
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Implement masks to detect road lines
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype = 'uint8')
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 175, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    # Add a gaussian blur
    kernel_size = 5
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)

    # Canny edge detection
    low_threshold = 80
    high_threshold = 150
    frame = cv2.Canny(gauss_gray, low_threshold, high_threshold)    

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
vs.release()
cv2.destroyAllWindoes()
