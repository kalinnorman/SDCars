"""
demo_steering.py
For testing our lane following algorithm.
"""
from CarControl import CarControl
import time
import random
import cv2

def wait_for_yellow_lane(cc):
    cc.rf.set_l_found(False)
    while not cc.rf.get_l_found():
        cc.update_sensors()  # update the sensors every loop
        t, rgb = cc.sensor.get_rgb_data()  # get color image
        frame, commands = cc.rf.find_lanes(rgb)  # find lines in image

def turn_handler(cc):
    current_region = cc.sensor.region
    if current_region == 'south':
        cc.action.turn_left_while_moving()
    elif current_region == 'middle south':
        cc.action.turn_right_while_moving()
    elif current_region == 'north':
        cc.action.turn_left_while_moving()
    elif current_region == 'middle north':
        cc.action.turn_right_while_moving()
    elif current_region == 'middle':
        direction = (random.randint(1,3) % 3)
        if ( direction == 0):
            cc.action.drive_straight()
        elif(direction == 1):
            cc.action.turn_left_while_moving()
        else:
            cc.action.turn_right_while_moving()
    else:
        print("I haven't a clue where I am.")



if __name__ == '__main__':

    cc = CarControl()  # create object to control car
    count = 0  # debouncer for finding limit lines
    speed = 0.25 # 0.3
#    list_of_actions = [0.0, 0.0, 0.0, 0.0, 0.0]

    # run the loop, waiting for a keyboard interrupt
    try:
        cc.steer(0)  # straighten steering
        lastSteerAngle = 0  # to keep track of steering value
        cc.drive(0.7)  # drive fast to get the car going
        time.sleep(0.5)  # get it up to speed
        cc.drive(speed)  # slow down to a slower speed
    
        count = 0
        while True:
            count += 1
            cc.update_sensors()  # update the sensors every loop
            t, rgb = cc.sensor.get_rgb_data()  # get color image
            
            object_detected = cc.detector.detect_object()
            if (object_detected):
                cc.drive(0.0)
                print('Object Detected!')
                continue
            else:
                cc.steer(0)  # straighten steering
                lastSteerAngle = 0  # to keep track of steering value
                cc.drive(0.7)  # drive fast to get the car going
                time.sleep(0.5)  # get it up to speed
                cc.drive(speed)  # slow down to a slower speed

            frame, commands = cc.rf.find_lanes(rgb, show_images=True)  # find lines in image

            # Parse outputs
            speed = commands[0]
            angle = commands[1]
            steering_state = commands[2]
            limit_found = commands[3]

            nextSteerAngle = angle 

           

            # Update steering angle
            """ FIXME 
            Look into making a queue and having it delay so it doesn't turn early?
            """
            if nextSteerAngle != lastSteerAngle:  # if the angle has changed
                cc.steer(nextSteerAngle)  # send the command
                lastSteerAngle = nextSteerAngle  # update the old value

            # # Handling intersections and corners
            """ FIXME
            Why is count > 75...?
            """
            if limit_found and count > 75:
               turn_handler(cc) 
               count = 0

            # Wait
            """ FIXME 
            Should this delay be adjusted? 
            """
            key = cv2.waitKey(25) & 0xFF  # wait a titch before the next loop

    except KeyboardInterrupt:  # when the user ctrl-C's the script
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print(cc.rf.get_counts())  # report the lane stats
        print("\nUser stopped the script. (KeyboardInterrupt)")

# We're done!
# Great job, car!
