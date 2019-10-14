# Top Level organization for navigation of autonomous car
# Takes in an image and identifies the proper power and angle
# IN TESTING - 10 Oct 2019!


# Make a LaneFollower() class
# Wrap lane follower in a try block 

import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import image as mpimg  # Unnecessary ???
import cv2
import math
from car_control import carControl


# Hashtag-Define's
leftColorMin = [90, 245, 180]        # Yellow - Determined by plotting imgHSV and hovering over the colors
leftColorMax = [100, 255, 210]       # Yellow
rightColorMin = [5, 15, 170]         # White
rightColorMax = [20, 40, 230]        # White
runNavigation = True

croppedHeightRatio = (1.0/2.0)
minSlope = 0.4
carCenter = 772     # X value of center of the car, as camera is offcenter


# def processImage(rawImg): 
#     imgHeight = rawImg.shape[0] 
#     imgWidth = rawImg.shape[1] 

#     # Clean image. May not actually do anything
#     kernel = np.ones((5,5), np.uint8)       # Not exactly sure how to pick values for this, this was just a recommendation I found...
#     cleanImg = cv2.morphologyEx(rawImg, cv2.MORPH_OPEN, kernel)
#     cleanImg = cv2.cvtColor(cleanImg, cv2.COLOR_BGR2RGB)

#     imgHSV = cv2.cvtColor(cleanImg, cv2.COLOR_BGR2HSV) # Convert to HSV

#     leftColorMin = np.asarray(leftColorMin)
#     leftColorMax = np.asarray(leftColorMax)
#     leftLineImg = cv2.inRange(imgHSV, leftColorMin, leftColorMax)

#     rightColorMin = np.asarray(rightColorMin)
#     rightColorMax = np.asarray(rightColorMax)
#     rightLineImg = cv2.inRange(imgHSV, rightColorMin, rightColorMax)

#         # Canny
#     leftCannyed = cv2.Canny(leftLineImg, 100, 200)          # Lower and upper thresholds chosen somewhat arbitrarily 
#     rightCannyed = cv2.Canny(rightLineImg, 100, 200)

#     return [leftCannyed, rightCannyed]

def calculateAngle(c, p1, p2):                    # AKA Inner Product
    c = np.asarray(c)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
                                            # (P1 dot P2) / |P1| * |P2|
    angle = np.arccos( (np.dot(p1, p2)) / 
                        (np.linalg.norm(p1) * np.linalg.norm(p2)) )
    
    return angle

def regionOfInterest(img, vertices):
    # Blank matrix that matches the image height/width
    mask = np.zeros_like(img)

    match_mask_color = 255 # Set to 255 to account for grayscale
    # channel_count = img.shape[2] # Number of color channels      -> Same as below
    # match_mask_color = (255,) * channel_count # Matches color    -> Commented out for grayscale

    cv2.fillPoly(mask, vertices, match_mask_color) # Fill polygon

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    tNumerator   = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    tDenominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    t = tNumerator/tDenominator

    intX = x1 + t * (x2 - x1)
    intY = y1 + t * (y2 - y1)
    
    intPoint = (int(intX), int(intY))
    return intPoint

def updateController(rightProcessed, leftProcessed):
    # Crop
    cropVertices = [(0, imgHeight),               # Corners of cropped image
                    (0, imgHeight * croppedHeightRatio),        # Gets bottom third
                    (imgWidth, imgHeight * croppedHeightRatio),
                    (imgWidth, imgHeight) ] 

    leftCropped = regionOfInterest(leftProcessed, np.array([cropVertices], np.int32))
    rightCropped = regionOfInterest(rightProcessed, np.array([cropVertices], np.int32))

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
    lspace = np.linspace(0, imgHeight, 10)
    drawY = lspace
    drawX = np.polyval(leftLine, drawY)       # May cause a problem if not a real function
    leftPoints = (np.asarray([drawX, drawY]).T).astype(np.int32)
    finalLeft = cv2.polylines(rawImg, [leftPoints], False, (255, 0, 225), thickness = 10)

    # Calculate and draw right line
    rightLine = np.polyfit(rightLine_y, rightLine_x, 1)
    drawX = np.polyval(rightLine, drawY)
    rightPoints = (np.asarray([drawX, drawY]).T).astype(np.int32)
    finalImg = cv2.polylines(finalLeft, [rightPoints], False, (255, 0, 255), thickness = 10)

    # Calculate intersection point
    intPoint = findIntersection(leftPoints[0][0], leftPoints[0][1],
                                leftPoints[1][0], leftPoints[1][1],
                                rightPoints[0][0], rightPoints[0][1],
                                rightPoints[1][0], rightPoints[1][1])

    # Draw line from center to intersection point
    centerPoint = (carCenter, imgHeight)
    cv2.line(finalImg, (centerPoint), (intPoint[0], intPoint[1]), (0,255,0), 5)

    # Calculate angle off of center
    midPoint = (carCenter, int(imgHeight/2))              # Center of screen. Used as a vertical reference for angle of departure
    cv2.line(finalImg, (centerPoint), (midPoint), (100, 200, 200), 4)
    angle = calculateAngle(centerPoint, intPoint, midPoint)

    angle = 180/np.pi

    return [angle, finalImg]

car = carControl()
car.drive(0.7)

while (runNavigation):
    # Get image 

    # path = 'C:/Users/benjj/Documents/College/Fall2019/Ecen522/Code/SDCars/class_code/image.jpg'
    # rawImg = cv2.imread(path)


    car.update_sensors()
    time, rawImg = car.get_rgb_data()

    imgHeight = rawImg.shape[0] 
    imgWidth = rawImg.shape[1] 

    # Clean image. May not actually do anything
    kernel = np.ones((5,5), np.uint8)       # Not exactly sure how to pick values for this, this was just a recommendation I found...
    cleanImg = cv2.morphologyEx(rawImg, cv2.MORPH_OPEN, kernel)
    cleanImg = cv2.cvtColor(cleanImg, cv2.COLOR_BGR2RGB)

    imgHSV = cv2.cvtColor(rawImg, cv2.COLOR_BGR2HSV) # Convert to HSV


    leftColorMin = np.asarray(leftColorMin)
    leftColorMax = np.asarray(leftColorMax)
    leftLineImg = cv2.inRange(imgHSV, leftColorMin, leftColorMax)

    rightColorMin = np.asarray(rightColorMin)
    rightColorMax = np.asarray(rightColorMax)
    rightLineImg = cv2.inRange(imgHSV, rightColorMin, rightColorMax)

        # Canny
    leftCannyed = cv2.Canny(leftLineImg, 100, 200)          # Lower and upper thresholds chosen somewhat arbitrarily 
    rightCannyed = cv2.Canny(rightLineImg, 100, 200)

    # [rightProcessed, leftProcessed] = processImage(frame)
    [angle, finalImg] = updateController(rightCannyed, leftCannyed)

    if angle > 30:
        angle = 29
    elif angle < -30:
        angle = -29

    car.steer(angle)

    # print(angle)

    # plt.imshow(finalImg)
    # plt.show()
