"""
Parameter file for road_detection.py
Author: redd
"""

import cv2

# The file to process
prefix = 'straight_road_1'
suffix = '.jpg'
file = prefix + suffix
output_file = prefix + '_output' + suffix

# Sets the threshold for blue channel
CHANNEL_THRESHOLD = 160

# Value to show all channels
ALL_CHANNELS = -1

# BGR channel values
BLUE_CHANNEL = 0
GREEN_CHANNEL = 1
RED_CHANNEL = 2

# HSV channel values
HUE_CHANNEL = 0
SATURATION_CHANNEL = 1
VALUE_CHANNEL = 2

# Edge finding default values
EDGE_LOW_THRESHOLD_DEFAULT = 350
EDGE_HIGH_THRESHOLD_DEFAULT = 450
EDGE_TYPE = None
EDGE_APERTURE_DEFAULT = 3
EDGE_L2GRADIENT = cv2.HOUGH_GRADIENT

# Line detection
LINE_HOUGH_THRESHOLD = 50

# Misc Parameters
LANE_INDICATION_COLOR = (0, 0, 255)  # color to add to image
