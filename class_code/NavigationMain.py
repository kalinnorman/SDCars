# Top Level organization for navigation of autonomous car
# Takes in an image and identifies the proper power and angle
# IN TESTING - 10 Oct 2019!


import numpy as np 
from matplotlib import pyplot as plt 
from LaneFollower import LaneFollower
import cv2
import math
from car_control import carControl


# Hashtag-Defines
runNavigation = True

car = carControl()
driveSpeed = 0.7
car.drive(driveSpeed)
lanef = LaneFollower()

while (runNavigation):
    # Get image 
    car.update_sensors()
    time, rawImg = cars.sensors.get_rgb_data()
    lanef.update_picture(rawImg)

    img = LF.convert_to_HSV(rawImg)

    leftMask = LF.filter_by_color(img, True)
    rightMask = LF.filter_by_color(img, False)

    leftCanny = LF.canny_img(leftMask)
    rightCanny = LF.canny_img(rightMask)

    leftCropped = LF.crop_image(leftCanny)
    rightCropped = LF.crop_image(rightCanny)

    leftLines = LF.hough_lines(leftCropped)
    rightLines = LF.hough_lines(rightCropped)

    try:
        left_line_x, left_line_y, right_line_x, right_line_y = LF.find_lines(leftLines, rightLines)
        leftFinal, left_points = LF.calculate_lines(rawImg, left_line_x, left_line_y, 1)
        rightFinal, right_points = LF.calculate_lines(leftFinal, right_line_x, right_line_y, 1)
        frame = rightFinal

        int_point = LF.find_intersection(left_points[0][0], left_points[0][1],
                                left_points[1][0], left_points[1][1],
                                right_points[0][0], right_points[0][1],
                                right_points[1][0], right_points[1][1])
        frame = LF.plot_center(frame, int_point)

        angle = LF.calculate_angle(int_point)
        driveSpeed = 0.7
        car.drive(driveSpeed)

    except:
        frame = frame
        driveSpeed = driveSpeed*.9
        car.drive(driveSpeed)

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    count = count + 1
    time.sleep(0.05)

    car.steer(angle)

