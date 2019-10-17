"""
Lane detection and following algorithm
"""
import cv2


class ReddFollower:

    def __init__(self):
        # do nothing
        1

    def filter_bright(self, frame):
        framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        framethresh = cv2.inRange(cv2.extractChannel(framehsv, 2), 175, 255)

        frameb = cv2.extractChannel(frame, 0)
        frameg = cv2.extractChannel(frame, 1)
        framer = cv2.extractChannel(frame, 2)

        newframe = frame.copy()
        newframe[:, :, 0] = cv2.bitwise_and(frameb, framethresh)
        newframe[:, :, 1] = cv2.bitwise_and(frameg, framethresh)
        newframe[:, :, 2] = cv2.bitwise_and(framer, framethresh)

        return frame, newframe

    def find_edges(self, frame):
        return cv2.Canny(frame, 50, 200)

    def separate_white_yellow(self, frame):
        frameb = cv2.extractChannel(frame, 0)
        framebthresh = cv2.inRange(frameb, 128, 255)
        return framebthresh, frame

    def find_lanes(self, frame):
        frame, newframe = self.filter_bright(frame)
        white, yellow = self.separate_white_yellow(newframe)
        frame = self.find_edges(newframe)
        return frame, white

