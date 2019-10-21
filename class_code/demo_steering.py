"""
demo_steering.py
For testing our lane following algorithm.
"""

from CarControl import CarControl
import time
import cv2

def wait_for_yellow_lane(cc):
    cc.rf.set_l_found(False)
    while not cc.rf.get_l_found():
        cc.update_sensors()  # update the sensors every loop
        t, rgb = cc.sensor.get_rgb_data()  # get color image
        frame, commands = cc.rf.find_lanes(rgb)  # find lines in image

if __name__ == '__main__':

    cc = CarControl()  # create object to control car
    count = 0  # debouncer for finding limit lines
    speed = 0.4 # 0.3

    # run the loop, waiting for a keyboard interrupt
    try:
        print("Beginning loop")
        cc.steer(0)  # straighten steering
        lastSteerAngle = 0  # to keep track of steering value
        cc.drive(0.6)  # drive fast to get the car going
        time.sleep(0.5)  # get it up to speed
        cc.drive(speed)  # slow down to a slower speed
        
        intersection_counter = 0
        corner_turn = 'r'
        type_of_turn = 'i'
        count = 0
        while True:
            count += 1
            cc.update_sensors()  # update the sensors every loop
            t, rgb = cc.sensor.get_rgb_data()  # get color image

            frame, commands = cc.rf.find_lanes(rgb, show_images=True)  # find lines in image

            # Parse outputs
            speed = commands[0]
            angle = commands[1]
            steering_state = commands[2]
            limit_found = commands[3]
            nextSteerAngle = angle

            # Update steering angle
            if nextSteerAngle != lastSteerAngle:  # if the angle has changed
                cc.steer(angle)  # send the command
                lastSteerAngle = nextSteerAngle  # update the old value

            # # Handling intersections and corners
            if limit_found and count > 75:
                print("I found the limit line!")
                count = 0
                if type_of_turn == 'i':
                    intersection_counter += 1
                    if intersection_counter % 3 == 1:
                        print('going straight through the intersection')
                        cc.action.drive_straight()
                        if corner_turn == 'l':
                            corner_turn = 'r'
                        else:
                            corner_turn = 'l'
                    elif intersection_counter % 3 == 2:
                        print('going left through the intersection')
                        cc.action.turn_left_while_moving()
                    else:
                        print('going right through the intersection')
                        cc.action.turn_right_while_moving()
                    type_of_turn = 'c'
                else:
                    if corner_turn == 'l':
                        print('turning left at a corner')
                        cc.action.turn_left_while_moving()
                    else:
                        print('turning right at a corner')
                        cc.action.turn_right_while_moving()
                wait_for_yellow_lane(cc)

           # For testing right turns only
#             if limit_found and count > 75:
#                 print("I found the limit line!")
#                 cc.action.turn_left_while_moving()
#                 count = 0
# #                cc.drive(speed)
#                 print('I found the yellow line and am done turning')
            
            # # For testing left turns only
            # if limit_found and count > 75:
            #     print("I found the limit line!")
           # #     cc.action.turn_left_while_moving()
            
            # # For testing only going straight through the intersection
            # if limit_found and count > 75:
            #     print("I found the limit line!")
            #     cc.action.drive_straight()

            # Show image
            # cv2.imshow('birds', frame)  # show the birdseye view

            # Wait
            key = cv2.waitKey(25) & 0xFF  # wait a titch before the next loop

    except KeyboardInterrupt:  # when the user ctrl-C's the script
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print(cc.rf.get_counts())  # report the lane stats
        print("\nUser stopped the script. (KeyboardInterrupt)")

# We're done!
# Great job, car!
