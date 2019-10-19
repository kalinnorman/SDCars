# ChooseTurn.py

# this file tells us what region the car is in, not sure if we will keep this as
# a class or put it inside another class, but whatever
# -ABT

import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import coordinates as co

# x range: 0-500
# y range: 0-1300
# ... I think
# note: values are inverted from what one would think actually makes sense
# (0,0) is top right corner
# x is vertical
# y is horizontal

class ChooseTurn:

    def __init__(self):
        self.x_max = 500
        self.y_max = 1300
        self.south = 'south'
        self.middleSouth = 'middleSouth'
        self.middle = 'middle'
        self.middleNorth = 'middleNorth'
        self.north = 'north'

    # call this when car has reached an intersection
    def getCoordinates(self):
        x, y = co.getCoor("Blue")  # outputs coordinates (x,y)

        # arbitrary values, need to test when have testing space
        if y > 1200:
            region = self.north
            # should go left
            print("north")
        elif y > 900:
            region = self.middleNorth
            # should go right
            print("middle north")
        elif y < 100:
            region = self.south
            # should go left
            print("south")
        elif y < 300:
            region = self.middleSouth
            # should go right
            print("middle south")
        else:
            region = self.middle
            # can go any direction
            print("middle")

        return region
