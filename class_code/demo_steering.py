"""
demo_steering.py
For testing our lane following algorithm.
"""

from CarControl import CarControl
import time
import cv2


if __name__ == '__main__':

    cc = CarControl()  # create object to control car
    count = 0  # debouncer for finding limit lines

    # run the loop, waiting for a keyboard interrupt
    try:
        print("Beginning loop")
        cc.steer(0)  # straighten steering
        lastSteerAngle = 0  # to keep track of steering value
        cc.drive(0.6)  # drive fast to get the car going
        time.sleep(0.5)  # get it up to speed
        cc.drive(0.35)  # slow down to a slower speed
        
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

            # Handling intersection
            if limit_found and count > 25:
                print("I found the limit line!")
                current_region = cc.sensor.region
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

            # Show image
            cv2.imshow('birds', frame)  # show the birdseye view

            # Wait
            key = cv2.waitKey(25) & 0xFF  # wait a titch before the next loop

    except KeyboardInterrupt:  # when the user ctrl-C's the script
        cc.drive(0.0)  # stop the car
        cc.steer(0.0)  # return wheels to normal
        print(cc.rf.get_counts())  # report the lane stats
        print("\nUser stopped the script. (KeyboardInterrupt)")

# We're done!
# Great job, car!
