"""
A class to detect roads in an image
Author: redd
"""

import cv2
import numpy as np
import road_detection_params as rdp
import color_detection as cd


class RoadDetection:
    """
    This class will read an image and return the image after finding the road
    """

    def __init__(self):
        """
        Constructor
        """
        # create a bunch of variables to store information about the detection
        self.img = 0  # stores the original image
        self.imggray = 0  # stores grayscale version of image
        self.imgedges = 0  # stores the edges of the image
        self.imglanes = 0  # stores the lanes of the image
        self.imgHSV = 0  # stores the image in HSV

        self.imgloaded = False  # indicates that an image has been loaded
        self.imgedgescalcd = False  # indicates that the image edges have been calculated
        self.lanesdetected = False  # indicates that lanes have been detected

    def load_image(self, frame, imgFromRealSense=False):
        """
        Loads the image given a string
        """
        self.img = frame  # read the image in

        if imgFromRealSense:
            self.remove_dark_bands()

        self.imglanes = cv2.copyTo(self.img, None, dst=None)
        self.imggray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.imgHSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.imgloaded = True

    def get_image(self):
        """
        Returns the image to the user.
        """
        return self.img

    def show_image(self, wait=True):
        """
        Shows the image to the user
        If wait == true, the execution will stop and wait for the user to press a key.
        Otherwise, the image will disappear if execution stops.
        """
        cv2.imshow('Unprocessed image', self.img)
        if wait:
            cv2.waitKey(0)

    def show_brg(self, channel=rdp.ALL_CHANNELS, wait=True):
        """
        Shows the BGR image to the user
        :param channel:
        :param wait:
        :return:
        """
        if channel == rdp.ALL_CHANNELS:
            cv2.imshow('All channels', self.img)
        elif channel == rdp.BLUE_CHANNEL:
            z = cv2.extractChannel(self.img, channel, dst=None)
            cv2.imshow('Blue channel', z)
        elif channel == rdp.GREEN_CHANNEL:
            z = cv2.extractChannel(self.img, channel, dst=None)
            cv2.imshow('Green channel', z)
        elif channel == rdp.RED_CHANNEL:
            z = cv2.extractChannel(self.img, channel, dst=None)
            cv2.imshow('Red channel', z)

        if wait:
            cv2.waitKey(0)

    def show_hsv(self, channel=rdp.ALL_CHANNELS, wait=True):
        """
        Shows the HSV image to the user.
        channel will indicate which channel to show.
        :param channel:
        :param wait:
        :return:
        """
        if channel == rdp.ALL_CHANNELS:
            # print('Choose a channel, bro.')
            z = cv2.extractChannel(self.imgHSV, rdp.HUE_CHANNEL, dst=None)
            cv2.imshow('Hue channel', z)
            z = cv2.extractChannel(self.imgHSV, rdp.SATURATION_CHANNEL, dst=None)
            cv2.imshow('Saturation channel', z)
            z = cv2.extractChannel(self.imgHSV, rdp.VALUE_CHANNEL, dst=None)
            cv2.imshow('Value channel', z)
        elif channel == rdp.HUE_CHANNEL:
            z = cv2.extractChannel(self.imgHSV, channel, dst=None)
            cv2.imshow('Hue channel', z)
        elif channel == rdp.SATURATION_CHANNEL:
            z = cv2.extractChannel(self.imgHSV, channel, dst=None)
            cv2.imshow('Saturation channel', z)
        elif channel == rdp.VALUE_CHANNEL:
            z = cv2.extractChannel(self.imgHSV, channel, dst=None)
            cv2.imshow('Value channel', z)

        if wait:
            cv2.waitKey(0)

    def find_edges(self,
                   lowThreshold=rdp.EDGE_LOW_THRESHOLD_DEFAULT,
                   highThreshold=rdp.EDGE_HIGH_THRESHOLD_DEFAULT,
                   edges=None,
                   apertureSize=rdp.EDGE_APERTURE_DEFAULT,
                   L2gradient=cv2.HOUGH_GRADIENT):
        """
        Finds the edges of the unprocessed image
        Must have already loaded an image
        :param lowThreshold:
        :param highThreshold:
        :param edges:
        :param apertureSize:
        :param L2gradient:
        :return:
        """
        if self.imgloaded:
            self.imgedges = cv2.Canny(self.imggray,
                                      lowThreshold,
                                      highThreshold,
                                      edges=edges,
                                      apertureSize=apertureSize,
                                      L2gradient=L2gradient)
            self.imgedgescalcd = True
        else:
            print('Please load an image.')

    def show_edges(self, wait=True):
        """
        Shows the edges of the image to the user.
        :param wait:
        :return:
        """
        if ~self.imgedgescalcd:  # If the edges haven't been calculated
            self.find_edges()  # Find the edges first

        if self.imgedgescalcd:  # if the edges have now been calculated
            cv2.imshow('Image Edges', self.imgedges)  # display the image
            if wait:
                cv2.waitKey(0)

    def find_lanes(self):
        """
        Finds the lanes in the image
        :return:
        """
        if self.imgedgescalcd:
            # Find lines
            # theta values:
            # 0 corresponds to vertical
            # pi/4 corresponds to diagonal from lower left-hand corner to upper right-hand corner
            # pi/3 corresponds to horizontal line
            lines = cv2.HoughLines(self.imgedges, 2, np.pi/180, 50, min_theta=0.7*np.pi/4, max_theta=1.3*np.pi/4, )

            # The below for loop runs till r and theta values
            # are in the range of the 2d array
            for i in range(0, len(lines)-1):
                for r, theta in lines[i]:
                    a = np.cos(theta)           # Stores the value of cos(theta) in a
                    b = np.sin(theta)           # Stores the value of sin(theta) in b
                    x0 = a * r                  # x0 stores the value rcos(theta)
                    y0 = b * r                  # y0 stores the value rsin(theta)
                    x1 = int(x0 + 1000 * (-b))  # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                    y1 = int(y0 + 1000 * (a))   # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                    x2 = int(x0 - 1000 * (-b))  # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                    y2 = int(y0 - 1000 * (a))   # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))

                    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                    # (0,0,255) denotes the colour of the line to be
                    # drawn. In this case, it is red.
                    cv2.line(self.imglanes, (x1, y1), (x2, y2), (0, 0, 255), 2)

            self.lanesdetected = True

        else:
            self.find_edges()

    def show_lanes(self, wait=True):
        """
        Shows the lanes of the image
        :param wait:
        :return:
        """
        if ~self.lanesdetected:
            self.find_lanes()

        if self.lanesdetected:
            cv2.imshow('Lanes', self.imglanes)
            if wait:
                cv2.waitKey(0)

    def find_straight_road(self, file, imgFromRealSense=False):
        """
        Straight road detection.
        Finds straight roads in the image.
        First displays the lines found for the left-hand side and the average of those lines.
        Then displays the lines found for the right-hand side and the average of those lines.
        :param file:
        :param imgFromRealSense:
        :return:
        """
        # load the image
        self.load_image(file, imgFromRealSense=imgFromRealSense)

        # look at blue channel
        imgB = cv2.extractChannel(self.img, rdp.BLUE_CHANNEL, dst=None)

        # Threshold the blue channel
        imgBt = cv2.inRange(imgB, 0, rdp.CHANNEL_THRESHOLD, dst=None)

        # Take bottom half of image
        imgBts = imgBt[int(imgBt.shape[0] / 2):(imgBt.shape[0]), 0:(imgBt.shape[1])]  # first index contains y values, second index contains x values

        # Calculate edges
        imgBtse = cv2.Canny(imgBts,
                            rdp.EDGE_LOW_THRESHOLD_DEFAULT,
                            rdp.EDGE_HIGH_THRESHOLD_DEFAULT,
                            edges=rdp.EDGE_TYPE,
                            apertureSize=rdp.EDGE_APERTURE_DEFAULT,
                            L2gradient=rdp.EDGE_L2GRADIENT)

        # Find lines
        # theta values:
        # 0 corresponds to vertical
        # pi/4 corresponds to diagonal from lower left-hand corner to upper right-hand corner
        # pi/2 corresponds to horizontal line
        leftlines = cv2.HoughLines(imgBtse, 2, np.pi / 180, rdp.LINE_HOUGH_THRESHOLD, min_theta=0.1 * np.pi / 4, max_theta=1.9 * np.pi / 4)

        leftline = np.mean(leftlines, 0)  # takes average of all lines found
        leftline = np.mean(leftline, 0)  # leftline is a list in a list, so this gets rid of the outer list

        rightlines = cv2.HoughLines(imgBtse, 2, np.pi / 180, rdp.LINE_HOUGH_THRESHOLD, min_theta=2.1 * np.pi / 4, max_theta=3.9 * np.pi / 4)

        rightline = np.mean(rightlines, 0)  # takes average of all lines found
        rightline = np.mean(rightline, 0)  # rightline is a list in a list, so this gets rid of the outer list

        # Show left lane boundary
        imgdummy = cv2.copyTo(self.img, None, dst=None)
        # self.show_lines_on_image(imgdummy, np.mean(leftlines, 1), offset=imgBtse.shape[0])
        self.show_line_on_image(self.imglanes, leftline[0], leftline[1], offset=imgBtse.shape[0])
        blank_image_1 = np.zeros(shape=self.imglanes.shape, dtype=np.uint8)
        self.show_line_on_image(blank_image_1, leftline[0], leftline[1], offset=imgBtse.shape[0])

        # Show right lane boundary
        imgdummy = cv2.copyTo(self.img, None, dst=None)
        # self.show_lines_on_image(imgdummy, np.mean(rightlines, 1), offset=imgBtse.shape[0])
        self.show_line_on_image(self.imglanes, rightline[0], rightline[1], offset=imgBtse.shape[0])
        blank_image_2 = np.zeros(shape=self.imglanes.shape, dtype=np.uint8)
        self.show_line_on_image(blank_image_2, rightline[0], rightline[1], offset=imgBtse.shape[0])

        # Find vanishing point
        intercept = cv2.bitwise_and(blank_image_1, blank_image_2)  # finds the intersection of the two lines
        non_zero = np.mean(np.mean(cv2.findNonZero(cv2.cvtColor(intercept, cv2.COLOR_BGR2GRAY)), 0), 0)  # calculate midpoint of intersection

        # Add trajectory line to image
        cv2.line(self.imglanes, (int(self.imglanes.shape[1]/2), int(self.imglanes.shape[0])), (int(non_zero[0]), int(non_zero[1])), (0, 255, 0), thickness=2)

        # Save the image with the lane indicators
        print("Saving output image.")
        cv2.imwrite(rdp.output_file, self.imglanes)
        print("Output image saved.")

        # Show images
        cv2.imshow('Line', self.imglanes)

        # Wait for user to press a key
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Indicate that the lanes were found
        self.lanesdetected = True

    def find_straight_road_2(self, frame, imgFromRealSense=False):
        """
        A modified algorithm ffor finding straight roads.
        :param file:
        :param imgFromRealSense:
        :return:
        """
        # load the image
        self.load_image(frame, imgFromRealSense=imgFromRealSense)

        # Take bottom half of image
        imghalf = self.img[int(self.img.shape[0] / 2):(self.img.shape[0]), 0:(self.img.shape[1])]  # first index contains y values, second index contains x values

        # Find white and yellow channels
        imgy = cd.detect_color(imghalf, "yellow")
        #imgy = cd.detect_hue(imghalf, "yellow")
        imgw = cd.detect_color(imghalf, "white")
        #imgw = cd.detect_hue(imghalf, "white")

        # Calculate edges
        imgye = cv2.Canny(imgy,
                            rdp.EDGE_LOW_THRESHOLD_DEFAULT,
                            rdp.EDGE_HIGH_THRESHOLD_DEFAULT,
                            edges=rdp.EDGE_TYPE,
                            apertureSize=rdp.EDGE_APERTURE_DEFAULT,
                            L2gradient=rdp.EDGE_L2GRADIENT)
        imgwe = cv2.Canny(imgw,
                            rdp.EDGE_LOW_THRESHOLD_DEFAULT,
                            rdp.EDGE_HIGH_THRESHOLD_DEFAULT,
                            edges=rdp.EDGE_TYPE,
                            apertureSize=rdp.EDGE_APERTURE_DEFAULT,
                            L2gradient=rdp.EDGE_L2GRADIENT)


        try:
            # Find left lane marker
            leftlines = cv2.HoughLines(imgye, 2, np.pi / 180, 50, min_theta=0.5 * np.pi / 4, max_theta=1.5 * np.pi / 4)
            leftline = np.mean(leftlines, 0)  # takes average of all lines found
            leftline = np.mean(leftline, 0)  # leftline is a list in a list, so this gets rid of the outer list

            # Find right lane marker
            rightlines = cv2.HoughLines(imgwe, 2, np.pi / 180, 80, min_theta=2.5 * np.pi / 4, max_theta=3.5 * np.pi / 4)
            rightline = np.mean(rightlines, 0)  # takes average of all lines found
            rightline = np.mean(rightline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            # Add lines to image
            # Show left lane boundary
            imgdummy = cv2.copyTo(self.img, None, dst=None)
            #self.show_lines_on_image(imgdummy, np.mean(leftlines, 1), offset=imgye.shape[0])
            self.show_line_on_image(self.imglanes, leftline[0], leftline[1], offset=imgye.shape[0])
            blank_image_1 = np.zeros(shape=self.imglanes.shape, dtype=np.uint8)
            self.show_line_on_image(blank_image_1, leftline[0], leftline[1], offset=imgye.shape[0])

            # Show right lane boundary
            imgdummy = cv2.copyTo(self.img, None, dst=None)
            #self.show_lines_on_image(imgdummy, np.mean(rightlines, 1), offset=imgwe.shape[0])
            self.show_line_on_image(self.imglanes, rightline[0], rightline[1], offset=imgwe.shape[0])
            blank_image_2 = np.zeros(shape=self.imglanes.shape, dtype=np.uint8)
            self.show_line_on_image(blank_image_2, rightline[0], rightline[1], offset=imgwe.shape[0])

            # Find vanishing point
            intercept = cv2.bitwise_and(blank_image_1, blank_image_2)  # finds the intersection of the two lines
            non_zero = np.mean(np.mean(cv2.findNonZero(cv2.cvtColor(intercept, cv2.COLOR_BGR2GRAY)), 0),
                               0)  # calculate midpoint of intersection

            # Add trajectory line to image
            cv2.line(self.imglanes, (int(self.imglanes.shape[1] / 2), int(self.imglanes.shape[0])),
                     (int(non_zero[0]), int(non_zero[1])), (0, 255, 0), thickness=2)

            # Show images
            cv2.imshow('Line', self.imglanes)

        except:
            cv2.imshow('Line', self.imglanes)

        # self.lanesdetected = True

    def show_lines_on_image(self, img, lines, offset=0):
        """
        Shows the user a set of lines on the image
        :param img:
        :param lines:
        :param offset:
        :return:
        """
        for line in lines:
            r = line[0]
            theta = line[1]

            a = np.cos(theta)  # Stores the value of cos(theta) in a
            b = np.sin(theta)  # Stores the value of sin(theta) in b
            x0 = a * r  # x0 stores the value rcos(theta)
            y0 = b * r  # y0 stores the value rsin(theta)
            x1 = int(x0 + 1000 * (-b))  # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            y1 = int(y0 + 1000 * (a))  # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            x2 = int(x0 - 1000 * (-b))  # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            y2 = int(y0 - 1000 * (a))  # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(img, (x1, y1 + offset), (x2, y2 + offset), rdp.LANE_INDICATION_COLOR,
                     2)

            cv2.imshow('Lines', img)

    def show_line_on_image(self, img, r, theta, offset=0):
        """
        Adds a single line to the image
        :param img:
        :param r:
        :param theta:
        :param offset:
        :return:
        """
        a = np.cos(theta)  # Stores the value of cos(theta) in a
        b = np.sin(theta)  # Stores the value of sin(theta) in b
        x0 = a * r  # x0 stores the value rcos(theta)
        y0 = b * r  # y0 stores the value rsin(theta)
        x1 = int(x0 + 1000 * (-b))  # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        y1 = int(y0 + 1000 * (a))  # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        x2 = int(x0 - 1000 * (-b))  # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        y2 = int(y0 - 1000 * (a))  # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        cv2.line(img, (x1, y1 + offset), (x2, y2 + offset), rdp.LANE_INDICATION_COLOR, 2)

        cv2.imshow('Line', img)

    def remove_dark_bands(self):
        """
        Removes the dark bands from the image from the RealSense camera
        :return:
        """
        border_width = 60
        self.img = self.img[border_width:self.img.shape[0] - border_width, 0:self.img.shape[1]]

    def horizontal_segment_image(self, img, num_segments, seg_number):
        """
        Segments an image into a number of horizontal images, and returns the n-th segment from the top.
        :param num_segments: The number of horizontal segments to divide image into
        :param seg_number: The O-indexed index of the image to return.
        :return:
        """
        y_offset = int(img.shape[0] / num_segments)
        y_start_coord = seg_number * y_offset
        y_end_coord = y_offset + y_start_coord
        return img[y_start_coord:y_end_coord, 0:(img.shape[1])]

    def curved_road(self, file, imgFromRealSense=False):
        """
        Algorithm to calculate where a curved road is in the image.
        :param file:
        :param imgFromRealSense:
        :return:
        """
        # load the image
        self.load_image(file, imgFromRealSense=imgFromRealSense)

        # Take bottom half of image
        imgh = self.horizontal_segment_image(self.img, 2, 1)  # first index contains y values, second index contains x values
        cv2.imshow('Bottom half', imgh)

        # Detect yellow/white
        imgyh = cd.detect_hue(imgh, "yellow", val_thresh=50, sat_thresh=130)

        # Edge detection
        imgyhe = cv2.Canny(imgyh, rdp.EDGE_LOW_THRESHOLD_DEFAULT,
                            rdp.EDGE_HIGH_THRESHOLD_DEFAULT,
                            edges=rdp.EDGE_TYPE,
                            apertureSize=rdp.EDGE_APERTURE_DEFAULT,
                            L2gradient=rdp.EDGE_L2GRADIENT)

        # Segment image
        number_of_segments = 6
        for i in range(0, number_of_segments):
            image_slice = self.horizontal_segment_image(imgyhe, number_of_segments, i)

            # Find left lane marker
            leftlines = cv2.HoughLines(image_slice, 2, np.pi / 180, 50, min_theta=0.5 * np.pi / 4, max_theta=1.5 * np.pi / 4)
            leftline = np.mean(leftlines, 0)  # takes average of all lines found
            leftline = np.mean(leftline, 0)  # leftline is a list in a list, so this gets rid of the outer list

            # Find right lane marker
            rightlines = cv2.HoughLines(image_slice, 2, np.pi / 180, 40, min_theta=2.5 * np.pi / 4, max_theta=3.5 * np.pi / 4)
            rightline = np.mean(rightlines, 0)  # takes average of all lines found
            rightline = np.mean(rightline, 0)  # rightline is a list in a list, so this gets rid of the outer list

            # Draw curve on image
            y1 = 0
            y2 = image_slice.shape[0]
            x2 = rightline[0] * np.cos(rightline[1])
            alpha = y2 / np.tan(np.pi/2 - rightline[1])
            x1 = alpha + x2

            self.show_line_on_image(image_slice, rightline[0], rightline[1])

            y_offset = i * image_slice.shape[0] + int(self.imglanes.shape[0] / 2)

            cv2.line(self.imglanes, (0, 0), (int(x2), int(y2) + y_offset), (0, i*50, i*50))
            cv2.line(self.imglanes, (int(x1), int(y1) + y_offset), (int(x2), int(y2) + y_offset), (0, 255, 0))

        cv2.imshow('Lanes', self.imglanes)

    def curved_road_2(self, file, imgFromRealSense=False):
        """
        Algorithm to calculate where a curved road is in the image.
        :param file:
        :param imgFromRealSense:
        :return:
        """
        # load the image
        self.load_image(file, imgFromRealSense=imgFromRealSense)
        cv2.imshow('Full image', self.img)

        # Take bottom half of image
        #imgh = self.horizontal_segment_image(self.img, 2, 1)  # first index contains y values, second index contains x values
        imgh = self.img[130:self.img.shape[0], 0:(self.img.shape[1])]
        cv2.imshow('Bottom half', imgh)

        # Detect yellow/white
        imgyh = cd.detect_hue(imgh, "yellow", val_thresh=50, sat_thresh=130)

        # Edge detection
        imgyhe = cv2.Canny(imgyh, rdp.EDGE_LOW_THRESHOLD_DEFAULT,
                           rdp.EDGE_HIGH_THRESHOLD_DEFAULT,
                           edges=rdp.EDGE_TYPE,
                           apertureSize=rdp.EDGE_APERTURE_DEFAULT,
                           L2gradient=rdp.EDGE_L2GRADIENT)
        imgyhe = cv2.blur(imgyhe, (3, 3))
        cv2.imshow('edges', imgyhe)

        #imgyhec = cv2.cvtColor(imgyhe, cv2.COLOR_GRAY2BGR)
        imgyhec = np.zeros((imgyhe.shape[0],imgyhe.shape[1],3), np.uint8)

        # Hough Lines P
        lines = cv2.HoughLinesP(imgyhe, 10, np.pi/180, 200, lines=None, minLineLength=3)
        #print(lines)

        x_pts = []
        y_pts = []

        rt_x_pts = []
        rt_y_pts = []

        lf_x_pts = []
        lf_y_pts = []

        for seg in lines:
            x_pts.append(seg[0][0])
            x_pts.append(seg[0][2])
            y_pts.append(seg[0][1])
            y_pts.append(seg[0][3])

            if seg[0][0] > 630:
                rt_x_pts.append(seg[0][0])
                rt_x_pts.append(seg[0][2])
                rt_y_pts.append(seg[0][1])
                rt_y_pts.append(seg[0][3])

                cv2.line(imgyhec, (seg[0][0], seg[0][1]), (seg[0][2], seg[0][3]), (0, 255, 255))
            elif (seg[0][0] > 460) & (seg[0][1] > 100):
                lf_x_pts.append(seg[0][0])
                lf_x_pts.append(seg[0][2])
                lf_y_pts.append(seg[0][1])
                lf_y_pts.append(seg[0][3])

                cv2.line(imgyhec, (seg[0][0], seg[0][1]), (seg[0][2], seg[0][3]), (255, 255, 0))

        cv2.imshow('segments', imgyhec)

        curve = np.polyfit(rt_y_pts, rt_x_pts, 2)
        lspace = np.linspace(0, 300, 100)
        draw_y = lspace
        draw_x = np.polyval(curve, draw_y)
        right_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        right_points_img = (np.asarray([draw_x, draw_y+130]).T).astype(np.int32)
        cv2.polylines(imgyhec, [right_points], False, (0, 150, 225), thickness = 3)
        cv2.polylines(self.imglanes, [right_points_img], False, (255, 0, 225), thickness = 3)

        curve = np.polyfit(lf_y_pts, lf_x_pts, 3)
        lspace = np.linspace(0, 300, 100)
        draw_y = lspace
        draw_x = np.polyval(curve, draw_y)
        left_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        left_points_img = (np.asarray([draw_x, draw_y + 130]).T).astype(np.int32)
        cv2.polylines(imgyhec, [left_points], False, (255, 0, 225), thickness = 3)
        cv2.polylines(self.imglanes, [left_points_img], False, (255, 0, 225), thickness=3)

        avg_x_pts = []
        avg_y_pts = []

        for i in range(0, len(right_points)):
            a = left_points[i]
            b = right_points[i]
            avg_x_pts.append((a[0] + b[0]) / 2)
            avg_y_pts.append((a[1] + b[1]) / 2)

        curve = np.polyfit(avg_y_pts, avg_x_pts, 4)
        lspace = np.linspace(0, 300, 100)
        draw_y = lspace
        draw_x = np.polyval(curve, draw_y)
        avg_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        avg_points_img = (np.asarray([draw_x, draw_y + 130]).T).astype(np.int32)
        cv2.polylines(imgyhec, [avg_points], False, (255, 0, 225), thickness = 3)
        cv2.polylines(self.imglanes, [avg_points_img], False, (127, 255, 127), thickness=3)

        cv2.imshow('curve', imgyhec)
        cv2.imshow('Final Result', self.imglanes)

        # save the image
        cv2.imwrite('curved_demo.jpg', self.imglanes)


# File to load
# file = rdp.file  # takes file specified in road_detection_params.py

# Create instance of RoadDetection
# rd = RoadDetection()

# Find the straight road
# rd.find_straight_road(file)

# More general algorithm
# rd.find_straight_road_2('real_straight.jpg', imgFromRealSense=True)

# Curved algorithm
# rd.curved_road('real_straight.jpg', imgFromRealSense=True)
# rd.curved_road('real_curve_2.jpg', imgFromRealSense=True)
# rd.curved_road_2('curved_highway_1.jpg', imgFromRealSense=False)

# cv2.waitKey(0)
