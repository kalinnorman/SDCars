"""
Script for recording, reading, and editing videos.

Author: redd
"""

import cv2
import numpy as np


def create_video(filename='output.avi'):
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:  # if the frame has been successfully captured
            # frame = cv2.flip(frame,0)  # flip the frame

            # write the frame
            out.write(frame)

            # show the video to the user
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def read_video(filename='output.avi'):
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def modify_video(filename='output.avi'):
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray[0:img.shape[0], 0:img.shape[1]] = img

        cv2.imshow('frame', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



"""
img = np.zeros([100, 100], np.uint8)

for i in range(1, 100):
    img[:, i] = i

cv2.imshow("watermark", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

modify_video()
"""