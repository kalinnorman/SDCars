"""
virtual_steering.py

Author: redd
"""
from CarControl import CarControl
import time
import cv2
import numpy as np


filename='Oct4DriveTest2.avi'

if __name__ == '__main__':

    print("Running virtual_steering.py")

    cap = cv2.VideoCapture(filename)

    cc = CarControl()  # create object to control car


    # run the loop, waiting for a keyboard interrupt
    try:
        time.sleep(1)
        print("Beginning loop")

        while True:

            if not cap.isOpened():
                raise Exception(KeyboardInterrupt)

            ret, frame = cap.read()

            frame, commands = cc.rf.find_lanes(frame)

            cv2.imshow('frame', frame)
            print(commands)
            cv2.waitKey(30)

            #time.sleep(1/30)  # for 30 ish fps

    except KeyboardInterrupt:
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        cap.release()
        cv2.destroyAllWindows()
        print("\nUser stopped the script. (KeyboardInterrupt)")

    cc.drive(0.0)  # stop the car
    cc.steer(0.0)  # return wheels to normal
    cap.release()
    cv2.destroyAllWindows()