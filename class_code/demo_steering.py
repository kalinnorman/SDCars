"""
demo_steering.py
Author: redd

This is a demo function for controlling the car
"""

import master_code as master
import time


def demo_steering():
    """
    Demo steering commands.
    :return:
    """
    print("Driving")
    master.steer(1200)
    time.sleep(1)
    master.steer(1800)
    time.sleep(1)
    master.drive(1600)
    time.sleep(2)
    master.drive(1500)
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