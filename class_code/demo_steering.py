"""
demo_steering.py

Author: redd

This is a demo function for controlling the car.
If this script this run as the main, this will initialize the car and tell it to turn the wheels and drive.
"""

import car_control as cc
import time


def steering_commands():
    """
    Demo steering commands.
    :return:
    """
    print("right")
    cc.steer(15.0)
    time.sleep(2)
    print("left")
    cc.steer(-15.0)
    time.sleep(2)
    print("go")
    cc.drive(0.5)
    time.sleep(2)
    print("stop")
    cc.drive(0.0)
    time.sleep(2)


if __name__ == '__main__':

    # initialize communication with the Arduino
    cc.initialize_car(pid=False)

    # run the loop, waiting for a keyboard interrupt
    try:
        print("Beginning loop")
        while True:
            steering_commands()  # run the sequence of steering commands
    except KeyboardInterrupt:
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print("User stopped the script. (KeyboardInterrupt)")
