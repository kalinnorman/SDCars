import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("saved_frame.jpg")

"""
Tunable Parameters
"""
warped_size = (200, 200)  # the size of the resultant warped image

car_perspective_y_top = 240
car_perspective_y_bottom = 480

car_perspective_x_bottom_left = 122
car_perspective_x_top_left = 283
car_perspective_x_top_right = 456
car_perspective_x_bottom_right = 700


"""
Calculations
"""
# Widths of lanes at various distances
lane_width_far = (car_perspective_x_top_right - car_perspective_x_top_left)
lane_width_close = (car_perspective_x_bottom_right - car_perspective_x_bottom_left)

# initial perspective
car_perspective = np.zeros((4, 2), dtype="float32")
car_perspective[0] = (car_perspective_x_bottom_left - 2*lane_width_close, car_perspective_y_bottom)
car_perspective[1] = (car_perspective_x_top_left - 2*lane_width_far, car_perspective_y_top)
car_perspective[2] = (car_perspective_x_top_right + 2*lane_width_far, car_perspective_y_top)
car_perspective[3] = (car_perspective_x_bottom_right + 2*lane_width_close, car_perspective_y_bottom)

# bird's-eye perspective
birds_eye = np.zeros((4, 2), dtype="float32")
birds_eye[0] = (0, warped_size[1])  # initial perspective
birds_eye[1] = (0, 0)
birds_eye[2] = (warped_size[0], 0)
birds_eye[3] = (warped_size[0], warped_size[1])

# Transformation Matrix
M = cv2.getPerspectiveTransform(car_perspective, birds_eye)  # calculate warping matrix

# Save transformation matrix to apply in other scripts
np.save('car_perspective_transform_matrix', M)

# Calculate new image
warped = cv2.warpPerspective(img, M, warped_size)  # perform the warping

"""
Show the before and after images
"""
plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(warped)
plt.show()
