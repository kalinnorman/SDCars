"""
Lane detection and following algorithm.

This class will
"""
import cv2
import numpy as np


class ReddFollower:

    def __init__(self):
        # do nothing
        self.steering_state = '.'  # '.' means don't turn, '<' means turn left, '>' means turn right
        self.car_control_speed = 0.4  # the commanded speed
        self.car_control_steering_angle = 0.0  # the commanded angle
        # self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix_short_range.npy')  # this matrix takes the car's perspective and transforms it to a bird's eye perspective
        # self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix_short_range_warp.npy')  # this matrix doesn't look as far forward
        self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix_warp_2.npy')  # this matrix accounts for the camera being consistently tilted
        self.theta_left_base = -0.5
        self.theta_right_base = -0.5
        self.center_pixel = 100
        self.right_desired_pixel = 113
        self.left_desired_pixel = 89
        # self.steer_factor = 4.0/3.0
        self.steer_factor = 1.0
        self.l_found = False
        self.counts = [0, 0, 0]  # for keeping track of number of frames where both lines, just the right, and just the left line are found.

    def filter_bright(self, frame):
        """
        Looks for the brightest colors in the images.
        Blacks out any part of the image that doesn't meet thoat threshold
        """
        # frameblur = cv2.blur(frame, (10, 10))
        framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert image to HSV
        framethresh = cv2.inRange(cv2.extractChannel(framehsv, 2), 175, 255)  # threshold based on value

        frameb = cv2.extractChannel(frame, 0)  # blue channel
        frameg = cv2.extractChannel(frame, 1)  # green channel
        framer = cv2.extractChannel(frame, 2)  # red channel

        # black out the portion of the image that doesn't meet the threshold
        newframe = frame.copy()  # create a new copy of the frame
        newframe[:, :, 0] = cv2.bitwise_and(frameb, framethresh)  # bitwise AND the blue channel and the thresholded image
        newframe[:, :, 1] = cv2.bitwise_and(frameg, framethresh)
        newframe[:, :, 2] = cv2.bitwise_and(framer, framethresh)

        return newframe  # returns the thresholded image

    def find_edges(self, frame, lowthresh=50, highthresh=200):
        """
        Find edges of the given image.
        """
        return cv2.Canny(frame, lowthresh, highthresh)

    def separate_white_yellow(self, frame):
        """
        Separates white and yellow components of image, kind of.
        That's what it's used for, but it basically white and not white.
        """
        frameb = cv2.extractChannel(frame, 0)
        white = cv2.inRange(frameb, 128, 255)
        notwhite = cv2.inRange(white, 0, 0)
        notblack = cv2.inRange(frameb, 1, 255)
        yellow = cv2.bitwise_and(notwhite, notblack)

        return white, yellow  # return the white part of the image and the yellow part of the image

    def find_limit_lines(self, white_edges):
        """
        looks in bottom quarter of image for horizontal lines that are limit lines.
        Pass in white edges image to work properly.
        Returns True if a limit line was found.
        Returns False if not.
        """
        try:  # try to find lines in the image
            high = white_edges.shape[0]
            low = int(3*high/4)
            white_edges_bottom_fourth = white_edges[low:high, :]
            limitlines = cv2.HoughLines(white_edges_bottom_fourth, 2, np.pi / 180, 200,
                                        min_theta=80 * np.pi / 180, max_theta =100 * np.pi / 180)
            limitline = np.mean(limitlines, 0)  # takes average of all lines found
            limitline = np.mean(limitline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            return True, limitline, low

        except:  # if we didn't find anything
            return False, (0, 0), 0  # return that the lane wasn't found

    def find_right_lane(self, frame):
        """
        Looks in bottom half of image for white lane
        Pass in the white edges image
        """
        try:  # try to find lines in the right half of the image
            # Blank out the left side of the image
            percentage_crop = .5  # we only want to look at the right half of the image
            width = int(frame.shape[1] * percentage_crop)
            black = np.zeros((frame.shape[0], width), "uint8")
            frame[:, 0:width] = black  # black out the left-hand side

            # Only look in bottom two-thirds of image
            high = frame.shape[0]  # height of image
            low = int(1/3*high)  # y-coordinate for a third of the way down
            white_edges_cropped = frame[low:high, :]  # crop the image

            rightlines = cv2.HoughLines(white_edges_cropped, 1, np.pi/180, 40,
                                        min_theta=-35*np.pi/180, max_theta=35*np.pi/180)
            rightline = np.mean(rightlines, 0)  # takes average of all lines found
            rightline = np.mean(rightline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            return True, rightline, low  # report that a line was found
        except:  # if a line wasn't found
            return False, (0, 0), 0  # report that nothing was found

    def find_left_lane(self, frame):
        """
        Looks in bottom half of image for yellow lane
        Pass in the yellow edges image
        """
        try: # try to find yellow lines
            # look in the bottom third of the image
            high = frame.shape[0]
            low = int(1/3*high)
            white_edges_cropped = frame[low:high, :]

            leftlines = cv2.HoughLines(white_edges_cropped, 1, np.pi / 180, 15,
                                        min_theta=-35 * np.pi / 180, max_theta=30 * np.pi / 180)
            leftline = np.mean(leftlines, 0)  # takes average of all lines found
            leftline = np.mean(leftline, 0)  # leftline is a list in a list, so this gets rid of the outer list

            return True, leftline, low  # report to user

        except:  # if nothing was found
            return False, (0, 0), 0  # report to user

    def steering_control(self, lane_parameters, print_info=False):
        """
        To control the steering
        """
        if lane_parameters[1] > self.theta_left_base:  # if the angle is to the right
            if self.steering_state != '>':  # if we haven't already
                self.steering_state = '>'  # tell the car to turn right
                if print_info:
                    print('turn right')  # inform the user
        else:  # if the angle is to the left
            if self.steering_state != '<':  # if we haven't already
                self.steering_state = '<'  # tell the car to turn left
                if print_info:
                    print('turn left')  # inform the user


    def find_lanes(self, frame, region=None, show_images=False):
        """
        The main function to call in this class
        """

        # Find limit line
        newframe = self.filter_bright(frame)  # looks for bright (white, yellow) colors in image
        white, yellow = self.separate_white_yellow(newframe)  # separates whites and yellows
        white_edges = self.find_edges(white)  # find white edges

        limit_found, limit_parameters, offset = self.find_limit_lines(white_edges)  # looks for horizonal limit lines

        # Find lanes
        birdseye_frame = cv2.warpPerspective(frame, self.birdseye_transform_matrix, (200, 200))  # transform to birdseye view
        newframe = self.filter_bright(birdseye_frame)  # looks for bright (white, yellow) colors in image
        white, yellow = self.separate_white_yellow(newframe)  # separates whites and yellows
        white_edges = self.find_edges(white)  # find white edges
        yellow_edges = self.find_edges(yellow)  # find yellow edges

        right_lane_found, right_parameters, right_offset = self.find_right_lane(white_edges)  # looks for right lane
        left_lane_found, left_parameters, left_offset = self.find_left_lane(yellow_edges)  # looks for right lane

        self.l_found = left_lane_found

        theta_deg_left = left_parameters[1]*(180/np.pi)
        theta_deg_right = right_parameters[1]*(180/np.pi)

        # print(theta_deg_left, theta_deg_right)
        # Show lines on images if desired
        # if left_lane_found:
        #     self.steering_control(left_parameters)

        if right_lane_found:  # if a right line is found
            rx1, ry1, rx2, ry2 = self.get_line_coordinates(birdseye_frame, right_parameters[0], right_parameters[1],
                                        offset=right_offset)  # get line coords
            x_r = self.get_x_intercept_bottom(rx1, ry1, rx2, ry2)

        if left_lane_found:  # if a left line is found
            lx1, ly1, lx2, ly2 = self.get_line_coordinates(birdseye_frame, left_parameters[0], left_parameters[1],
                                        offset=left_offset)  # get line coords
            x_l = self.get_x_intercept_bottom(lx1, ly1, lx2, ly2)

        if right_lane_found and left_lane_found:
            self.counts[0] += 1
            angle, centered = self.center_car_given_both_lanes(x_r, x_l)
            centered = True
            if centered:
                self.car_control_steering_angle = self.follow_both_lanes(theta_deg_right, theta_deg_left)
            else:
                self.car_control_steering_angle = angle
        if right_lane_found:
            self.counts[2] += 1
            angle, centered = self.center_car_given_right_lane(x_r)
            centered = True
            if centered:
                self.car_control_steering_angle = self.follow_right_lane(theta_deg_right)
            else:
                self.car_control_steering_angle = angle
            
        elif left_lane_found:
            self.counts[1] += 1
            angle, centered = self.center_car_given_left_lane(x_l)
            centered = True
            if centered:
                self.car_control_steering_angle = self.follow_left_lane(theta_deg_left)
            else:
                self.car_control_steering_angle = angle

        if self.car_control_steering_angle > 30:
            self.car_control_steering_angle = 30
        elif self.car_control_steering_angle < -30:
            self.car_control_steering_angle = -30
            

        # IMPORTANT
        # If a limit line is found, demo_steering.py will override the decisions made above
        control_values = (self.car_control_speed, self.car_control_steering_angle, self.steering_state, limit_found)

        # if show_images:
        if True:
            # if limit_found:  # if a limit line is found
            #     self.get_line_coordinates(frame, limit_parameters[0], limit_parameters[1],
            #                             offset=offset, showImg=True)  # draw it on the image

            if right_lane_found:  # if a right line is found
                self.get_line_coordinates(birdseye_frame, right_parameters[0], right_parameters[1],
                                        offset=right_offset, showImg=True)  # draw it on the image

            if left_lane_found:  # if a right line is found
                self.get_line_coordinates(birdseye_frame, left_parameters[0], left_parameters[1],
                                        offset=left_offset, showImg=True)  # draw it on the image
            #cv2.imshow('frame', frame)
            #cv2.imshow('misc', white_edges)
            # cv2.imshow('yellow', yellow_edges)
            #cv2.imshow('birdseye', birdseye_frame)

        # return frame, control_values  # returns original frame and commands
        return birdseye_frame, control_values  # returns the bird's eye view with lane indications and commands

    def get_line_coordinates(self, img, r, theta, offset=0, showImg=False):
        a = np.cos(theta)  # Stores the value of cos(theta) in a
        b = np.sin(theta)  # Stores the value of sin(theta) in b
        x0 = a * r  # x0 stores the value rcos(theta)
        y0 = b * r  # y0 stores the value rsin(theta)
        x1 = int(x0 + 1000 * (-b))  # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        y1 = int(y0 + 1000 * (a))  # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        x2 = int(x0 - 1000 * (-b))  # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        y2 = int(y0 - 1000 * (a))  # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        if showImg:
            cv2.line(img, (x1, y1 + offset), (x2, y2 + offset), (0, 0, 255), 2)

        return x1, y1+offset, x2, y2+offset

    def get_x_intercept_bottom(self, x1, y1, x2, y2, y0=200):
        return ((y0-y1)*(x2-x1))/(y2-y1) + x1

    def get_counts(self):
        """
        Returns the counts for the number of frames that contained both lanes, just the right, and just the left.
        """
        return self.counts

    def center_car_given_both_lanes(self, x_r, x_l, slope_r=999, slope_l=999):
        lane_center = round((x_r+x_l)/2.0)
        avg_slope = (slope_r+slope_l)/2.0
        # The function currently does not weight the turn based off of the slope, but it may be necessary
        # to change that depending on how well this function works on sharp turns
        dist_from_center = lane_center - self.center_pixel
        angle_to_steer = round((2.0/3.0)*dist_from_center)
        if abs(dist_from_center) <= 1:
            centered = True
        else:
            centered = False

        return angle_to_steer, centered

    def center_car_given_right_lane(self, x, slope=999):
        # Currently ignores slope, may need to modify behavior based on slope later
        dist_from_desired = self.right_desired_pixel - x
        angle_to_steer = round((2.0/3.0)*dist_from_desired)
        if abs(dist_from_desired) <= 2:
            centered = True
        else:
            centered = False

        return angle_to_steer, centered


    def center_car_given_left_lane(self, x, slope=999):
        # Currently ignores slope, may need to modify behavior based on slope later
        dist_from_desired = self.left_desired_pixel - x
        angle_to_steer = round((2.0/3.0)*dist_from_desired)
        if abs(dist_from_desired) <= 2:
            centered = True
        else:
            centered = False

        return angle_to_steer, centered

    def follow_both_lanes(self, theta_r, theta_l):
        avg_theta = (theta_r+theta_l)/2.0
        angle_to_steer = round(self.steer_factor*avg_theta)
        return angle_to_steer

    def follow_right_lane(self, theta):
        angle_to_steer = round(self.steer_factor*theta)
        return angle_to_steer

    def follow_left_lane(self, theta):
        angle_to_steer = round(self.steer_factor*theta)
        return angle_to_steer

    def get_l_found(self):
        return self.l_found
    
    def set_l_found(self, val):
        self.l_found = val