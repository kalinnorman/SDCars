# Testing new version of StraightLine.py
# Filters by color, uses yellow for left side and white for right side
# Created Oct 8, 2019

import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import image as mpimg
import cv2
import math

# Hashtag-Define's
leftColorMin = [90, 245, 180]        # Yellow - Determined by plotting imgHSV and hovering over the colors
leftColorMax = [100, 255, 210]       # Yellow
rightColorMin = [5, 15, 170]         # White
rightColorMax = [20, 40, 230]        # White

croppedHeightRatio = (2.0/3.0)
minSlope = 0.4

# Functions
def regionOfInterest(img, vertices):
    # Blank matrix that matches the image height/width
    mask = np.zeros_like(img)

    match_mask_color = 255 # Set to 255 to account for grayscale
    # channel_count = img.shape[2] # Number of color channels      -> Same as below
    # match_mask_color = (255,) * channel_count # Matches color    -> Commented out for grayscale

    cv2.fillPoly(mask, vertices, match_mask_color) # Fill polygon

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

##### 1) Load Image #####
path = 'C:/Users/benjj/Documents/College/Fall2019/Ecen522/TestingPhotos/Turn1.jpg'
rawImg = cv2.imread(path) 
imgHeight = rawImg.shape[0] 
imgWidth = rawImg.shape[1] 

# Clean Image   -    TESTING: Not sure if this actually helps

kernel = np.ones((5,5), np.uint8)       # Not exactly sure how to pick values for this, this was just a recommendation I found...
cleanImg = cv2.morphologyEx(rawImg, cv2.MORPH_OPEN, kernel)
cleanImg = cv2.cvtColor(cleanImg, cv2.COLOR_BGR2RGB)

imgHSV = cv2.cvtColor(cleanImg, cv2.COLOR_BGR2HSV) # Convert to HSV

##### 2) Filter #####

leftColorMin = np.asarray(leftColorMin)
leftColorMax = np.asarray(leftColorMax)
leftLineImg = cv2.inRange(imgHSV, leftColorMin, leftColorMax)

rightColorMin = np.asarray(rightColorMin)
rightColorMax = np.asarray(rightColorMax)
rightLineImg = cv2.inRange(imgHSV, rightColorMin, rightColorMax)

    # Canny
leftCannyed = cv2.Canny(leftLineImg, 100, 200)          # Lower and upper thresholds chosen somewhat arbitrarily 
rightCannyed = cv2.Canny(rightLineImg, 100, 200)

    # Crop
cropVertices = [(0, imgHeight),               # Corners of cropped image
                (0, imgHeight * croppedHeightRatio),        # Gets bottom third
                (imgWidth, imgHeight * croppedHeightRatio),
                (imgWidth, imgHeight) ] 

leftCropped = regionOfInterest(leftCannyed, np.array([cropVertices], np.int32))
rightCropped = regionOfInterest(rightCannyed, np.array([cropVertices], np.int32))

# Hough Transforms to find lines -> Note that leftLine arguments is more lenient
leftLines = cv2.HoughLinesP(
    leftCropped,
    rho = 6,
    theta = np.pi/60,
    threshold = 100,
    lines = np.array([]),
    minLineLength = 20,
    maxLineGap = 80)

rightLines = cv2.HoughLinesP(
    rightCropped,
    rho = 6,
    theta = np.pi/60,
    threshold = 150,
    lines = np.array([]),
    minLineLength = 60,
    maxLineGap = 40)


# Filter out lines and group remaining
leftLine_x = []
leftLine_y = []
rightLine_x = []
rightLine_y = []

for line in leftLines:
    for x1, y1, x2, y2 in line:
        slope = (y2-y1) / (x2-x1)
        if math.fabs(slope) > minSlope:          # Ignore lines with slopes that aren't what we are looking for
            leftLine_x.extend([x1, x2])          # Add valid lines to the list
            leftLine_y.extend([y1, y2])

for line in rightLines:
    for x1, y1, x2, y2 in line:
        slope = (y2-y1) / (x2 - x1)
        if ((math.fabs(slope) > minSlope) & (slope>=0)):
            rightLine_x.extend([x1, x2])         
            rightLine_y.extend([y1, y2])


# Calculate and draw left line
leftLine = np.polyfit(leftLine_y, leftLine_x, 1)
lspace = np.linspace(imgHeight * croppedHeightRatio, imgHeight, 10)
drawY = lspace
drawX = np.polyval(leftLine, drawY)       # May cause a problem if not a real function
leftPoints = (np.asarray([drawX, drawY]).T).astype(np.int32)
finalLeft = cv2.polylines(rawImg, [leftPoints], False, (255, 0, 225), thickness = 10)

# Calculate and draw right line
rightLine = np.polyfit(rightLine_y, rightLine_x, 1)
drawX = np.polyval(rightLine, drawY)
rightPoints = (np.asarray([drawX, drawY]).T).astype(np.int32)
final = cv2.polylines(finalLeft, [rightPoints], False, (255, 0, 255), thickness = 10)

plt.imshow(final)
plt.show()
