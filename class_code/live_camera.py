#live_camera.py

# import the necessary packages
import argparse
import imutils
import cv2	

vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
vs.release()
cv2.destroyAllWindoes()
