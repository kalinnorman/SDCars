"""
PredictiveFollower.py

TODO Possibly check for out-of-region points that could trick the search algorithm into trying to follow the gray value on the other side.
An alternative would be to simply change the map we are using between one for the inner regions and one for the outer regions.

Author: redd
"""
import cv2
import numpy as np


####################
# Helper functions #
####################
def show_scaled(img, scale_down_factor=3, wait=False, title="map"):
    """
    Shows a scaled version of a given image
    :param img: The image to show
    :param scale_down_factor: How much to scale it
    :param wait: If you want a cv2.waitKey(0)
    :param title: In case you have conflicting window names
    :return: nothing
    """
    display_dimensions = (img.shape[1] // scale_down_factor, img.shape[0] // scale_down_factor)
    global_map_small = cv2.resize(img, display_dimensions)
    cv2.imshow(title, global_map_small)
    if wait:
        cv2.waitKey(1)


def create_rotation_matrix(angle):
    """
    Creates a rotation matrix for a given angle in radians.
    :param angle: The angle to rotate in radians
    :return: the rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


####################
# Class Definition #
####################
class PredictiveFollower:

    def __init__(self, blurred_img, search_radius=70, desired_gray_val=215, number_of_dich_steps=10):
        """
        Initializes the function
        """
        # Map parameters
        self.blur = blurred_img  # loads the map (Should be in grayscale)

        # Search parameters
        self.desired_gray_val = desired_gray_val  # the value to track
        self.search_radius = search_radius  # how far ahead to look
        self.number_of_dich_steps = number_of_dich_steps  # number of iterations for dichotomous search algorithm

        # Display image parameters
        self.color_intensity = 255 / self.number_of_dich_steps  # intensity of colors on search plot
        self.demo = self.blur.copy()

    def find_angle(self, x_curr, y, x_prev, y_prev, show_plots=False):
        """
        Finds the angle in which the car should drive (radians)
        :param x_curr: current map x coordinate
        :param y: current map y coordinate
        :param x_prev: previous map x_curr coordinate
        :param y_prev: previous map y coordinate
        :param show_plots: Shows dichotomous search, if desired
        :return: the angle to steer in radians
        """

        angle = 0   # 0 represents straight in front of the car, +90 is to the right, -90 is to the left
        angle_step = 90 * np.pi / 180  # represents how much the dichotomous search algorithm will adjust after each iteration

        # If the previous and the current locations are the same,
        # We need to give the car a direction to orient itself
        if x_curr == x_prev and y == y_prev:  # this will prevent divide by zero errors
            k = np.array([[1], [0]])
        else:
            k = np.array([[x_curr - x_prev], [y - y_prev]])

        k = self.search_radius * k / np.linalg.norm(k)  # represents direction the car is looking
        rot_mat = create_rotation_matrix(0)  # rotation matrix for search

        if show_plots:
            self.demo = self.blur.copy()  # get a fresh copy of the map
            self.demo = cv2.cvtColor(self.demo, cv2.COLOR_GRAY2BGR)  # and convert it for pretty colors

        # Dichotomous Search
        for i in range(0, self.number_of_dich_steps):
            k = rot_mat @ k  # direction to look
            angle_step /= 2  # divide by two for next time

            # Create the point where to look to prevent indexing errors
            y_search = int(y + k[1])
            x_search = int(x_curr + k[0])
            if x_search > 1599 or x_search < 0:
                gray_value = 0
            elif y_search > 1023 or y_search < 0:
                gray_value = 0
            else:
                gray_value = self.blur[y_search, x_search]

            # update parameters for next time
            if gray_value < self.desired_gray_val:
                angle -= angle_step
                rot_mat = create_rotation_matrix(-angle_step)
            elif gray_value > self.desired_gray_val:
                angle += angle_step
                rot_mat = create_rotation_matrix(angle_step)
            else:
                return angle

            # Add history of which points were looked at and add them to the image
            if show_plots:
                cv2.circle(self.demo, (int(x_curr + k[0]), int(y + k[1])), 10, (0, i * self.color_intensity, 0), thickness=-1)

        # Show where the points we looked at were
        # Brighter indicates more accurate estimate
        if show_plots:
            show_scaled(self.demo, title="demo")
            cv2.waitKey(0)

        # Returns the angle in radians
        return angle
    
    def set_img(self,img):
        self.blur = img
    
    def set_gray_val(self, desired_gray_val):
        self.desired_gray_val = desired_gray_val

    def set_search_radius(self, radius):
        self.search_radius = radius

    def set_dichotomous_steps(self, steps):
        self.number_of_dich_steps = steps