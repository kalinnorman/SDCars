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
        cc.steer(0)
        lastSteerAngle = 0
        cc.drive(0.7)
        time.sleep(0.5)
        cc.drive(0.25)
        
        count = 0
        while True:
            count += 1
            cc.update_sensors()
            t, rgb = cc.sensor.get_rgb_data()  # get color image
            #t, depth = cc.sensor.get_depth_data()  # get depth data
            #depth_scaled = ((depth / np.median(depth)) * 128).astype(dtype='uint8')  # depth is super finicky
            #depth_scaled = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_AUTUMN)  # apply color map for pretty colors

            frame, commands = cc.rf.find_lanes(rgb, show_images=True)
            speed = commands[0]
            angle = commands[1]
            steering_state = commands[2]
            limit_found = commands[3]
            nextSteerAngle = angle
            if nextSteerAngle != lastSteerAngle:
                cc.steer(angle)
                lastSteerAngle = nextSteerAngle

            if limit_found and count > 25:
                print("I found the limit line!")
                current_region = cc.sensor.get_gps_region()
                if current_region == 'south':
                    print("I'm turning left at the southern stop sign.")
                    cc.action.turn_left_while_moving()
                elif current_region == 'middle south':
                    print("I'm turning right at the southern stop sign.")
                    cc.action.turn_right_while_moving()
                elif current_region == 'middle north':
                    print("I'm turning right at the northern stop sign.")
                    cc.action.turn_right_while_moving()
                elif current_region == 'north':
                    print("I'm turning left at the northern stop sign.")
                elif current_region == 'middle':
                    print("I'm at the intersection.")
                    if (count % 3) == 0:
                        print("I decided to go straight.")
                        cc.action.drive_straight()
                    elif (count % 3) == 1:
                        print("I decided to turn right.")
                        cc.action.turn_right_while_moving()
                    else:
                        print("I decided to turn left.")
                        cc.action.turn_left_while_moving()
                else:
                    print("I haven't a clue where I am.")

                count = 0
            cv2.imshow('birds', frame)
            # time.sleep(0.005)
            key = cv2.waitKey(25) & 0xFF

    except KeyboardInterrupt:
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print(cc.rf.get_counts())
        print("\nUser stopped the script. (KeyboardInterrupt)")

