"""
demo_steering.py

Author: redd

This is a demo function for controlling the car.
If this script this run as the main, this will initialize the car and tell it to turn the wheels and drive.
"""

from car_control import steer, drive, initialize_car
import time


def steering_commands():
    """
    Demo steering commands.
    :return:
    """
    steer(15.0)
    time.sleep(1)
    steer(-15.0)
    time.sleep(1)
    steer(0.0)
    drive(0.4)
    time.sleep(2)
    drive(0.5)
    time.sleep(2)
    drive(0)
    drive(0)
    time.sleep(1)


if __name__ == '__main__':

    # initialize communication with the Arduino
    initialize_car(pid=False)

    # run the loop, waiting for a keyboard interrupt
    try:
        while True:
            print("Loop")
            steering_commands()  # run the sequence of steering commands
    except KeyboardInterrupt:
        drive(0.0)  # stop the car
        steer(0.0)  # return wheels to normal
        print("User stopped the script. (KeyboardInterrupt)")
