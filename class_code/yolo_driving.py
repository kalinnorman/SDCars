'''
driving with yolo
'''

import cv2
from yolo_4_autumn import Yolo


def get_gray_value(coordinates, img): # Converts from cv2 coords to coords on Dr Lee's image
    imgWidth = img.shape[1] # Get width
    x = round(coordinates[0]) # x translates directly
    y = imgWidth - round(coordinates[1]) # y is inverted
    cur_gps = (x,y)
    gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
    cur_gray_val = gray_val
    return gray_val

def crop_image(img, box):
    """
    Takes in an image and crops out a specified range
    """
    cropVertices = [(box[0], box[1]),                      # Corners of cropped image
        (box[1], box[2]),        # Gets bottom portion
        (box[2], box[3]),
        (box[3], box[4]) ]


yo = Yolo()
# driving stuff

# before anything else, check that we're in the region for yolo
# if we are, check for yolo (have a counter for each frame so you aren't checking
# it at every frame)

yolo_map = cv2.imread('Maps/yolo_regions.bmp')
check_yolo_region = 123

cv2.circle(yolo_map, (600,300) ,5, color=(0,255,0))

#print(get_gray_value((600,300), yolo_map)[0])
if get_gray_value((300,600), yolo_map)[0] == check_yolo_region :
    print('DEBUG: oh yeah!')

# cv2.imshow('map', yolo_map)
# cv2.waitKey(0)

imgLight = cv2.imread('traffic_lights/red_light2.png')
cv2.rectangle(imgLight, (364, 161), (383, 194), (0,255,0))

cv2.imshow("light", imgLight)
cv2.waitKey(0)

# img_middle = int(imgLight.shape[1]/2)

# put in Sensors.py: self.yolo_count = 0
# if it's in the four regions, # look at Maps/regions.bmp
# run yolo
# check that traffic light is on the right side
# get bounding coordinates

img_middle = 208    # for some reason this isn't the same as half the camera size, but oh well!

coordinates = (300,600)
if get_gray_value(coordinates, yolo_map)[0] == check_yolo_region :
    # do yolo
    # get vector of bounding boxes from yolo and check that they're on the right side
    bounding_boxes = yo.main_yolo(imgLight)
    light_boxes = []
    # bounding_box = [x1, y1, x2, y2]
    # bounding_box = [229, 138, 232, 145]
    for box in range(0, len(bounding_boxes)):
        x1 = bounding_boxes[box][0]
        x2 = bounding_boxes[box][2]

        if x1 > img_middle and x2 > img_middle :  # bounding box is on the right side of the camera
            light_boxes.append(bounding_boxes[box])
            print (light_boxes[-1])
    if not light_boxes:
        break # exit frame and try again

    # figure out how to crop image and get color
    # check what color is inside it
    # what if there is more than one light being detected??
    # guess that the closer one will be bigger?
    # or whichever one is closest to the top of the image?
