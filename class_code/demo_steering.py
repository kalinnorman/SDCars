"""
demo_steering.py
Author: redd

This is a demo function for controlling the car
"""

from car_control import steer, drive
import time


def demo():
    """
    Demo steering commands.
    :return:
    """
    print("Driving")
    steer(1200)
    time.sleep(1)
    steer(1800)
    time.sleep(1)
    drive(1600)
    time.sleep(2)
    drive(1700)
    time.sleep(2)
    drive(1600)
    drive(1500)
    time.sleep(1)

    '''
    # Example car control
    print("Turn right")
    steer(1800)
    time.sleep(1)
    print("Turn left")
    steer(1200)
    time.sleep(1)
    print("Turn straight")
    steer(1500)
    time.sleep(1)

    print("Drive Forward")
    drive(1700)
    time.sleep(1)
    print("Stop")
    drive(1500)
    time.sleep(1)
    print("Drive Backward")
    drive(1300)
    time.sleep(1)
    print("Stop")
    drive(1500)
    time.sleep(1)

    '''