##### Testing

# from car_control import carControl
from LaneFollower import LaneFollower
from matplotlib import pyplot as plt
import numpy as np 
import cv2

# leftColorMin = np.asarray([90, 245, 180])        # Yellow - Determined by plotting imgHSV and hovering over the colors
# leftColorMax = np.asarray([100, 255, 210])       # Yellow
# rightColorMin = np.asarray([65, 60, 190])        # White   - Updated values
# rightColorMax = np.asarray([85, 80, 210])        # White

leftColorMin = np.asarray([50, 200, 150])        # Yellow - Determined by plotting imgHSV and hovering over the colors
leftColorMax = np.asarray([150, 255, 255])       # Yellow
rightColorMin = np.asarray([10, 20, 150])        # White   - Updated values
rightColorMax = np.asarray([110, 100, 255])        # White


# rightColorMin = np.asarray([5, 15, 170])         # White - Old values
# rightColorMax = np.asarray([20, 40, 230])        # White

LF = LaneFollower()
# car = carControl()
# car = carControl()
# car.drive(0.7)

path = 'C:/Users/benjj/Documents/College/Fall2019/Ecen522/TestingPhotos/Calibration_image.jpg'
rawImg = cv2.imread(path)

LF.update_picture(rawImg)
img = LF.clean_image(rawImg)

img = LF.convert_to_HSV(img)


leftMask = LF.filter_by_color(img, leftColorMin, leftColorMax)
rightMask = LF.filter_by_color(img, rightColorMin, rightColorMax)

leftCanny = LF.canny_img(leftMask)
rightCanny = LF.canny_img(rightMask)

leftCropped = LF.crop_image(leftCanny)
rightCropped = LF.crop_image(rightCanny)

image = leftCropped | rightCropped
plt.imshow(image)
plt.show()

leftLines = LF.hough_lines(leftCropped)
rightLines = LF.hough_lines(rightCropped)

left_line_x, left_line_y, right_line_x, right_line_y = LF.find_lines(leftLines, rightLines)

leftFinal, left_points = LF.calculate_lines(rawImg, left_line_x, left_line_y, 1)
rightFinal, right_points = LF.calculate_lines(leftFinal, right_line_x, right_line_y, 1)

intersection_point = LF.find_intersection(left_points[0][0], left_points[0][1],
                                        left_points[1][0], left_points[1][1],
                                        right_points[0][0], right_points[0][1],
                                        right_points[1][0], right_points[1][1])

final_image = LF.plot_center(rightFinal, intersection_point)

angle = LF.calculate_angle(intersection_point)

print(angle)
plt.imshow(final_image)
plt.show()