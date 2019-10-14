# IN TESTING - 10 Oct 2019!

# To do:
#   - Wrap in a try block, if fails have it go straight and slow 
#   - Tune parameters (Canny, Clean, etc...)
#   - Crop image
#   - Standardize coding style

import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

class LaneFollower:
    """
    Works with an image to find lanes and determine a steering anble
    """

    def __init__(self):
        self.leftColorMin = np.asarray([90, 245, 180])        # Yellow - Determined by plotting imgHSV and hovering over the colors
        self.leftColorMax = np.asarray([100, 255, 210])       # Yellow
        self.rightColorMin = np.asarray([5, 15, 170])         # White
        self.rightColorMax = np.asarray([20, 40, 230])        # White
        self.croppedHeightRatio = (1.0/2.0)     # Dimensions of the cropped image
        self.minSlope = 0.4      # Used to filter out lines that couldn't be the lanes
        self.max_angle = 30      # Maximum steering angle
            # NOT USED CURRENTLY #
        self.carCenter = 772     # X value of center of the car, as camera is offcenter

    def update_picture(self, img):
        """
        Takes in an image and updates the dimension values
        """
        self.raw_image = img                    # Stores a copy of the raw image
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
            # MAY NEED TO BE UPDATED TO CAR CENTER #
        self.center_point = (int(self.imgWidth/2), self.imgHeight) 

    def clean_image(self, img):
        """
        UNDER TESTING !!!
        Smooths out image. 
        Values need to be tuned
        """
        kernel = np.ones((5,5), up.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def convert_to_HSV(self, img):
        """
        Converts a RGB image to HSV
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def filter_by_color(self, imgHSV, color_min, color_max):
        """ 
        Searches through an HSV image and filters out all pixels not in 
        the specified range of values.
        """
        min_array = np.asarray([color_min])
        max_array = np.asarray([color_max])
        return cv2.inRange(imgHSV, min_array, max_array)
 
    def canny_img(self, img):
        """
        Finds all edges in an image
        """
        return cv2.Canny(img, 100, 200)     # Lower and upper thresholds not chosen for any specific reason

    def crop_image(self, img):
        """
        Takes in an image and crops out a specified range
        """
        cropVertices = [(0, self.imgHeight),                      # Corners of cropped image
            (0, self.imgHeight * self.croppedHeightRatio),        # Gets bottom portion
            (self.imgWidth, self.imgHeight * self.croppedHeightRatio),
            (self.imgWidth, self.imgHeight) ] 

        # Blank matrix that matches the image height/width
        mask = np.zeros_like(img)

        match_mask_color = 255 # Set to 255 to account for grayscale
        # channel_count = img.shape[2] # Number of color channels      -> Same as below
        # match_mask_color = (255,) * channel_count # Matches color    -> Commented out for grayscale

        cv2.fillPoly(mask, np.array([cropVertices], np.int32), match_mask_color) # Fill polygon

        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def find_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Takes in two points on two lines (4 points total)
        Determines the slope and then uses the given points to calculate 
            an intersection point.
        """

        t_numerator   = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t_denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        t = t_numerator/t_denominator

        intX = x1 + t * (x2 - x1)
        intY = y1 + t * (y2 - y1)

        int_point = (int(intX), int(intY))
        return int_point

    def hough_lines(self, img):
        """
        Takes in edges and connects the pixels into lines
        PARAMETERS CAN BE ADJUSTED
        """
        return cv2.HoughLinesP(
                img,
                rho = 6,
                theta = np.pi/60,
                threshold = 100,
                lines = np.array([]),
                minLineLength = 20,
                maxLineGap = 80)

    def find_lines(self, left_lines, right_lines):
        """
        Takes in a list of lines and sorts through to find the lanes
        """
        # Filter out lines and group remaining
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []


        for line in left_lines:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1) / (x2-x1)
                if math.fabs(slope) > self.minSlope:          # Ignore lines with slopes that aren't what we are looking for
                    left_line_x.extend([x1, x2])                # Add valid lines to the list
                    left_line_y.extend([y1, y2])

        for line in right_lines:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1) / (x2 - x1)
                if ((math.fabs(slope) > self.minSlope) & (slope>=0)):   # Ignore lines with slopes that are invalid
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        return left_line_x, left_line_y, right_line_x, right_line_y

    def calculate_lines(self, img, line_x, line_y, order):
        """
        Takes in a list of x and y coordinates and uses them to fit a line of a designated order
        """
        line = np.polyfit(line_y, line_x, 1)
        lspace = np.linspace(0, self.imgHeight, 10)     # Generates the line for the whole image height
        drawY = lspace
        drawX = np.polyval(line, drawY)                 # May cause a problem if not a real function
        points = (np.asarray([drawX, drawY]).T).astype(np.int32)                    # Points on line
        final = cv2.polylines(img, [points], False, (255, 0, 225), thickness = 10)  # Draws lines

        return final, points

    def plot_center(self, img, intersection_point):
        """
        Draws a line from the center of the screen to the intersection points of the lanes
        """
        # centerPoint = (self.carCenter, imgHeight)     # Used if we are accounting for the camera offset
        cv2.line(img, (self.center_point), (intersection_point[0], intersection_point[1]), (0,255,0), 5)

        return img

    def saturate(self, angle):
        """
        Takes in an angle and ensures it is within the permitted range
        """
        if (abs(angle) > self.max_angle):
            angle = self.max_angle * np.sign(angle)
        return angle

    def calculate_angle(self, intersection_point):                      # Uses inner product
        """
        Uses the intersection point, center point, and midpoint of the screen to calculate 
            the angle (uses an inner product)
        """

        # Determines the points and formats as arrays
        mid_point = (int(self.imgWidth/2), int(self.imgHeight/2))
        c = np.asarray(self.center_point)
        p1 = np.asarray(intersection_point)
        p2 = np.asarray(mid_point)    

        angle = np.arccos( (np.dot(p1, p2)) /           # (P1 dot P2) / |P1| * |P2|
                            (np.linalg.norm(p1) * np.linalg.norm(p2)) )

        angle = angle * 180/np.pi                       # Convert to degrees

        angle = self.saturate(angle)                    # Saturate angle
        return angle

