"""
demo_steering.py

Author: redd

This is a demo function for controlling the car.
If this script this run as the main, this will initialize the car and tell it to turn the wheels and drive.
"""

from CarControl import CarControl
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
    cc.drive(0.6)
    time.sleep(2)
    print("stop")
    cc.drive(0)
    time.sleep(2)
    

if __name__ == '__main__':

    cc = CarControl()  # create object to control car
    count = 0

    # run the loop, waiting for a keyboard interrupt
    try:
        time.sleep(1)
        print("Beginning loop")
        cc.drive(0.6)
        cc.steer(-3)
        while True:

            cc.update_sensors()
            t, rgb = cc.sensor.get_rgb_data()  # get color image
            #t, depth = cc.sensor.get_depth_data()  # get depth data
            #depth_scaled = ((depth / np.median(depth)) * 128).astype(dtype='uint8')  # depth is super finicky
            #depth_scaled = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_AUTUMN)  # apply color map for pretty colors

            frame, commands = cc.rf.find_lanes(rgb)
            
            speed = commands[0]
            angle = commands[1]
            steering_state = commands[2]
            limit_found = commands[3]

            #print((speed, angle*2))
            #if count == 5:
            #    cc.steer(angle*2)
            #    count = 0
            #else:
            #    count = count + 1
            #steering_commands()

            if limit_found:
                cc.action.turn_right_while_moving()
                print("I found the limit line!")

    except:
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print("\nUser stopped the script. (KeyboardInterrupt)")

