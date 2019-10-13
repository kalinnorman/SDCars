"""
demo_steering.py

Author: redd

This is a demo function for controlling the car.
If this script this run as the main, this will initialize the car and tell it to turn the wheels and drive.
"""

from car_control import CarControl
import time
import cv2
import numpy as np

def steering_commands():
    """
    Demo steering commands.
    :return:
    """
    print("right")
    cc.steer(15)
    time.sleep(2)
    print("left")
    cc.steer(-15.0)
    time.sleep(2)
    print("go")
    cc.drive(0.5)
    time.sleep(2)
    print("stop")
    cc.drive(0)
    time.sleep(2)


if __name__ == '__main__':

    cc = CarControl()  # create object to control car

    # run the loop, waiting for a keyboard interrupt
    try:
        time.sleep(1)
        print("Beginning loop")
        while True:

            cc.update_sensors()
            t, rgb = cc.get_rgb_data()  # get color image
            t, depth = cc.get_depth_data()  # get depth data
            depth_scaled = ((depth / np.median(depth)) * 128).astype(dtype='uint8')  # depth is super finicky
            depth_scaled = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_AUTUMN)  # apply color map for pretty colors

            print(t)
            #print(rgb)
            #print(depth)

            cv2.imwrite("color.jpg", rgb)  # show color image
            cv2.imwrite("depth.jpg", depth_scaled)  # show depth image

            steering_commands()  # run the sequence of steering commands

            cv2.destroyAllWindows()  # reset the windows for the next loop

            cc.action.drive_straight()  # drive straight for a bit.

    except KeyboardInterrupt:
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print("User stopped the script. (KeyboardInterrupt)")

