"""
Lane detection and following algorithm
"""
import cv2
import numpy as np


class ReddFollower:

    def __init__(self):
        # do nothing
        1

    def filter_bright(self, frame):
        """
        Looks for the brightest colors in the images
        """
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
            cv2.imshow('white_edges_stuff', white_edges_bottom_fourth)
            limitlines = cv2.HoughLines(white_edges_bottom_fourth, 2, np.pi / 180, 200,
                                        min_theta=80 * np.pi / 180, max_theta =100 * np.pi / 180)
            limitline = np.mean(limitlines, 0)  # takes average of all lines found
            limitline = np.mean(limitline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            return True, limitline
        except:
            # nothing found, don't do anything
            return False, (0, 0)


    def find_lanes(self, frame):
        """
        The main function to call in this class
        """
        newframe = self.filter_bright(frame)  # looks for bright (white, yellow) colors in image
        white, yellow = self.separate_white_yellow(newframe)  # separates whites and yellows

        white_edges = self.find_edges(white)  # find white edges
        yellow_edges = self.find_edges(yellow)  # find yellow edges

        limit_found, limit_parameters = self.find_limit_lines(white_edges)  # looks for horizonal limit lines

        if limit_found:  # if a limit line is found
            self.show_line_on_image(frame, limit_parameters[0], limit_parameters[1], offset=360)  # draw it on the image

        return frame, white_edges, yellow_edges  # return these images for plotting

    def show_line_on_image(self, img, r, theta, offset=0):
        a = np.cos(theta)  # Stores the value of cos(theta) in a
        b = np.sin(theta)  # Stores the value of sin(theta) in b
        x0 = a * r  # x0 stores the value rcos(theta)
        y0 = b * r  # y0 stores the value rsin(theta)
        x1 = int(x0 + 1000 * (-b))  # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        y1 = int(y0 + 1000 * (a))  # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        x2 = int(x0 - 1000 * (-b))  # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        y2 = int(y0 - 1000 * (a))  # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        cv2.line(img, (x1, y1 + offset), (x2, y2 + offset), (0, 0, 255), 2)

