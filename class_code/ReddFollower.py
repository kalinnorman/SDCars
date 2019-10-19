"""
Lane detection and following algorithm
"""
import cv2
import numpy as np


class ReddFollower:

    def __init__(self):
        # do nothing
        self.steering_state = '.'  # '.' means don't turn, '<' means turn left, '>' means turn right
        self.car_control_speed = 0.4
        self.car_control_steering_angle = 0.0
        self.birdseye_transform_matrix = np.load('car_perspective_transform_matrix.npy')
        self.theta_left_base = -0.5
        self.theta_right_base = -0.4

    def filter_bright(self, frame):
        """
        Looks for the brightest colors in the images
        """
        # frameblur = cv2.blur(frame, (10, 10))
        framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        framethresh = cv2.inRange(cv2.extractChannel(framehsv, 2), 175, 255)

        frameb = cv2.extractChannel(frame, 0)
        frameg = cv2.extractChannel(frame, 1)
        framer = cv2.extractChannel(frame, 2)

        newframe = frame.copy()
        newframe[:, :, 0] = cv2.bitwise_and(frameb, framethresh)
        newframe[:, :, 1] = cv2.bitwise_and(frameg, framethresh)
        newframe[:, :, 2] = cv2.bitwise_and(framer, framethresh)

        return newframe  # returns

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
        return white, yellow

    def find_limit_lines(self, white_edges):
        """
        looks in bottom quarter of image for horizontal lines that are limit lines.
        Pass in white edges image to work properly.
        Returns True if a limit line was found.
        Returns False if not.
        """
        try:
            # Find lines
            # theta values:
            # 0 corresponds to vertical
            # pi/4 corresponds to diagonal from lower left-hand corner to upper right-hand corner
            # pi/2 corresponds to horizontal line
            high = white_edges.shape[0]
            low = int(3*high/4)
            white_edges_bottom_fourth = white_edges[low:high, :]
            limitlines = cv2.HoughLines(white_edges_bottom_fourth, 2, np.pi / 180, 200,
                                        min_theta=80 * np.pi / 180, max_theta =100 * np.pi / 180)
            limitline = np.mean(limitlines, 0)  # takes average of all lines found
            limitline = np.mean(limitline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            return True, limitline, low
        except:
            # nothing found, don't do anything
            return False, (0, 0), 0

    def find_right_lane(self, white_edges):
        """
        Looks in bottom half of image for white lane
        Pass in the white edges image
        """
        try:

            percentage_crop = .4
            width = int(white_edges.shape[1] * percentage_crop)
            black = np.zeros((white_edges.shape[0], width), "uint8")
            white_edges[:, 0:width] = black
            rightlines = cv2.HoughLines(white_edges, 1, np.pi/180, 40,
                                        min_theta=-45*np.pi/180, max_theta=45*np.pi/180)
            rightline = np.mean(rightlines, 0)  # takes average of all lines found
            rightline = np.mean(rightline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            return True, rightline, 0
        except:
            # nothing found, don't do anything
            return False, (0, 0), 0

    def find_left_lane(self, frame):
        """
        Looks in bottom half of image for yellow lane
        Pass in the yellow edges image
        """
        try:
            # Find lines
            # theta values:
            # 0 corresponds to vertical
            # pi/4 corresponds to diagonal from lower left-hand corner to upper right-hand corner
            # pi/2 corresponds to horizontal line
            high = frame.shape[0]
            low = int(2/3*high)
            white_edges_bottom_fourth = frame[low:high, :]

            leftlines = cv2.HoughLines(white_edges_bottom_fourth, 1, np.pi / 180, 20,
                                        min_theta=-45 * np.pi / 180, max_theta =70 * np.pi / 180)
            leftline = np.mean(leftlines, 0)  # takes average of all lines found
            leftline = np.mean(leftline, 0)  # leftline is a list in a list, so this gets rid of the outer list

            return True, leftline, low
        except:
            # nothing found, don't do anything
            return False, (0, 0), 0

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


    def find_lanes(self, frame, show_images=False):
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

        theta_deg_left = left_parameters[1]*(180/np.pi)
        theta_deg_right = right_parameters[1]*(180/np.pi)

        print(theta_deg_left, theta_deg_right)
        # Show lines on images if desired
        if left_lane_found:
            self.steering_control(left_parameters)

        if right_lane_found:  # if a right line is found
            rx1, ry1, rx2, ry2 = self.get_line_coordinates(birdseye_frame, right_parameters[0], right_parameters[1],
                                        offset=right_offset)  # get line coords

        if left_lane_found:  # if a left line is found
            lx1, ly1, lx2, ly2 = self.get_line_coordinates(birdseye_frame, left_parameters[0], left_parameters[1],
                                        offset=left_offset)  # get line coords


        if right_lane_found and left_lane_found:
            print('both lanes')
            lastView = "both"
            # if theta_deg_left > 10.0:
            #     self.car_control_steering_angle = 14
            # elif theta_deg_left < -10.0:
            #     self.car_control_steering_angle = -14
            # else:
            #     if abs(theta_deg_right) > abs(theta_deg_left):
            #         self.car_control_steering_angle = -5
            #     elif abs(theta_deg_left) > abs(theta_deg_right):
            #         self.car_control_steering_angle = 5
            #     else:
            #         self.car_control_steering_angle = 0
            if self.steering_state == '<':
                if abs(theta_deg_left) > 10.0:
                    self.car_control_steering_angle = -15
                else:
                    self.car_control_steering_angle = -5
            elif self.steering_state == '>':
                if abs(theta_deg_left) > 10.0:
                    self.car_control_steering_angle = 15
                else:
                    self.car_control_steering_angle = 5
        elif left_lane_found:
            print('left lane')
            lastView = "left"
            if self.steering_state == '<':
                if abs(theta_deg_left) > 10.0:
                    self.car_control_steering_angle = -15
                else:
                    self.car_control_steering_angle = -5
            elif self.steering_state == '>':
                if abs(theta_deg_left) > 10.0:
                    self.car_control_steering_angle = 15
                else:
                    self.car_control_steering_angle = 5
        elif right_lane_found:
            print('right lane')
            lastView = "right"
            if theta_deg_right < self.theta_right_base:
                if abs(theta_deg_right) > 10.0:
                    self.car_control_steering_angle = -15
                else:
                    self.car_control_steering_angle = -5
            elif theta_deg_right >= self.theta_right_base:
                if abs(theta_deg_right) > 10.0:
                    self.car_control_steering_angle = 15
                else:
                    self.car_control_steering_angle = 5
        else:
            print('NO LANES FOUND!!!')
        # if abs(left_parameters[1]) > 0:
        #     if self.steering_state == '<':
        #         if theta_deg_left < -10.0:
        #             self.car_control_steering_angle = -19 # KALIN TRIED TO ADD SHARPER TURNS
        #         elif theta_deg_left < -8.0:
        #             self.car_control_steering_angle = -17
        #         elif theta_deg_left < -5.0:
        #             self.car_control_steering_angle = -11
        #         else:
        #             self.car_control_steering_angle = -7
        #     elif self.steering_state == '>':
        #         if theta_deg_left > 10.0:
        #             self.car_control_steering_angle = 13 # KALIN TRIED TO ADD SHARPER TURNS
        #         elif theta_deg_left > 8.0:
        #             self.car_control_steering_angle = 11
        #         elif theta_deg_left > 5.0:
        #             self.car_control_steering_angle = 5
        #         else:
        #             self.car_control_steering_angle = 1

        """     
        if (self.steering_state == '<') and (left_parameters[1] > 0):
            self.steering_state = '.'
            self.car_control_steering_angle = 0
        elif (self.steering_state == '>') and (left_parameters[1] < 0):
            self.steering_state = '.'
            self.car_control_steering_angle = 0
        """

        control_values = (self.car_control_speed, self.car_control_steering_angle, self.steering_state, limit_found)

        if show_images:
            if limit_found:  # if a limit line is found
                self.get_line_coordinates(frame, limit_parameters[0], limit_parameters[1],
                                        offset=offset, showImg=True)  # draw it on the image

            if right_lane_found:  # if a right line is found
                self.get_line_coordinates(birdseye_frame, right_parameters[0], right_parameters[1],
                                        offset=right_offset, showImg=True)  # draw it on the image

            if left_lane_found:  # if a right line is found
                self.get_line_coordinates(birdseye_frame, left_parameters[0], left_parameters[1],
                                        offset=left_offset, showImg=True)  # draw it on the image
            cv2.imshow('frame', frame)
            cv2.imshow('misc', white_edges)
            cv2.imshow('yellow', yellow_edges)
            cv2.imshow('birdseye', birdseye_frame)

        return frame, control_values  # return these images for plotting

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

