# GPSRegions.py
'''
This file tells us which of the 5 regions the car is in.

May or may not put this into another class or just add more to this class later
 -ABT
'''
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from Sensors import Sensors

# x range: 0-500
# y range: 0-1300
# ... I think
# note: values are inverted from what one would think actually makes sense
# (0,0) is top right corner
# x is vertical
# y is horizontal

s = Sensors()

class GPSRegions:

    def __init__(self):
        # values are not really necessary right now... but who knows!
        self.x_max = 500
        self.y_max = 1300

    # call this when car has reached an intersection
    def get_gps_region(self):
        x,y = s.get_gps_coord("Blue")  # outputs coordinates (x,y)

        # arbitrary values, need to test when have testing space
        if y > 1200 :
            region = 'north'
            # should go left
        elif y > 900 :
            region = 'middle north'
            # should go right
        elif y < 100 :
            region = 'south'
            # should go left
        elif y < 300 :
            region = 'middle south'
            # should go right
        else :
            region = 'middle'
            # can go any direction
        return region
