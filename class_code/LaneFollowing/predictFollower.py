import cv2
import numpy as np

# Helper Function
def create_rotation_matrix(angle):
    """
    Creates a rotation matrix for a given angle in radians.
    :param angle: The angle to rotate in radians
    :return: the rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

class PredictiveFollower:

    def __init__(self, input_map, search_radius=70, desired_gray_val=128, number_of_dich_steps=20):
        """
        Initializes the function
        """
        # Map parameters
        self.map = input_map  # loads the map (Should be in grayscale)

        # Search parameters
        self.desired_gray_val = desired_gray_val  # the value to track
        self.search_radius = search_radius  # how far ahead to look
        self.number_of_dich_steps = number_of_dich_steps  # number of iterations for dichotomous search algorithm

    def find_angle(self, x_curr, y, x_prev, y_prev):
        """
        Finds the angle in which the car should drive (radians)
        :param x_curr: current map x coordinate
        :param y: current map y coordinate
        :param x_prev: previous map x_curr coordinate
        :param y_prev: previous map y coordinate
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
                gray_value = self.map[y_search, x_search]

            # update parameters for next time
            if gray_value < self.desired_gray_val:
                angle -= angle_step
                rot_mat = create_rotation_matrix(-angle_step)
            elif gray_value > self.desired_gray_val:
                angle += angle_step
                rot_mat = create_rotation_matrix(angle_step)
            else:
                print("Found the desired Value!")
                return angle

        # Returns the angle in radians
        print("didnt...")
        return angle
    
    def set_img(self,img):
        self.map = img
    
    def set_gray_val(self, desired_gray_val):
        self.desired_gray_val = desired_gray_val

    def set_search_radius(self, radius):
        self.search_radius = radius

    def set_dichotomous_steps(self, steps):
        self.number_of_dich_steps = steps