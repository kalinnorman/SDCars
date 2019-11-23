# USAGE
# sudo MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 yolo.py
# OPTIONAL PARAMETERS
# -c/--confidence (.0-1.0) (detected objects with a confidence higher than this will be used)

# import the necessary packages
from CarControl import CarControl #steer, drive
from matplotlib import pyplot as plt
from gluoncv import model_zoo, utils
import pyrealsense2 as rs
from PIL import Image
from signal import signal, SIGINT
from sys import exit
import numpy as np
import mxnet as mx
import argparse
import imutils
# import serial
import time
import cv2
import os
import gc

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

"""Transforms for YOLO series."""
def transform_test(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)

        img = mx.nd.image.normalize(img, mean=mean, std=std)

        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs

def load_test(filenames, short=416):
    if not isinstance(filenames, list):
        filenames = [filenames]
    imgs = [letterbox_image(f, short) for f in filenames]
    return transform_test(imgs)


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size=416):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size

    scale = min(size/iw, size/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (size, size), (128, 128, 128))
    new_image.paste(image, ((size-nw)//2, (size-nh)//2))
    return mx.nd.array(np.array(new_image))

# Function to correctly exit program
def handler(signal_received, frame):
    vs.release()
    cv2.destroyAllWindows()
    print('CTRL-C detected. Exiting gracefully')
    exit(0)

# YOLO - added from Sensors.py
def find_color(img, color):
	# Convert image to HSV
	imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Define the desired colorspace
	if color == 'red':
		lower = np.array([120, 40, 40], dtype='uint8') # was [150, 40, 40]
		upper = np.array([255, 255, 255], dtype='uint8')
	elif color == 'green':
		lower = np.array([50, 40, 40], dtype='uint8')
		upper = np.array([100, 255, 255], dtype='uint8')
	elif color == 'yellow':
		lower = np.array([0, 40, 40], dtype='uint8')
		upper = np.array([50, 255, 255], dtype='uint8')
	else:
		print("Choose a valid color, bro.")

	# Threshold the HSV image to get only the desired color
	mask = cv2.inRange(imghsv, lower, upper)
	res = cv2.bitwise_and(img, img, mask=mask)
	count = cv2.countNonZero(res[:,:,0])
	#cv2.imshow('img', res)
	#cv2.waitKey(0)

	return res, count  # returns the image and the count of non-zero pixels

# YOLO - added from sensors
def predict_color(img):

	colors = ['red', 'yellow', 'green']
	counts = []

	for color in colors:
		res, count = find_color(img, color)
		counts.append(count)

	return colors[counts.index(max(counts))]  # returns the color as a string

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # ls -ltr /dev/video*
(W, H) = (None, None)

# Implement YOLOv3MXNet
net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

# Set device to GPU
device=mx.gpu()

net.collect_params().reset_ctx(device)

signal(SIGINT, handler)
print('Running. Press CTRL-C to exit')

current_bb = [0, 0, 0, 0]

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # from gluoncv import data
    yolo_image = Image.fromarray(frame, 'RGB')
    x, img = load_test(yolo_image, short=416)
    img_middle = 208

    class_IDs, scores, bounding_boxs = net(x.copyto(device))

    # The next two lines draw boxes around detected objects
    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
    plt.show()

    # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # plt.imshow(imgHSV)
    # plt.show()

    #print(class_IDs)
    #print(scores)
    #print(bounding_boxs)


    # Convert to numpy arrays, then to lists
    class_IDs = class_IDs.asnumpy().tolist()
    scores = scores.asnumpy().tolist()
    bounding_boxs = bounding_boxs.asnumpy()
    traffic_boxes = []
    # iterate through detected objects
    for i in range(len(class_IDs[0])):
        if ((scores[0][i])[0]) > args["confidence"]:
            current_class_id = net.classes[int((class_IDs[0][i])[0])]
            current_score = (scores[0][i])[0]
            current_bb = bounding_boxs[0][i-1]
            if current_class_id == 'traffic light':
                traffic_boxes.append(current_bb)
                # start of Sensor.py stuff
    light_boxes = []
    # bounding_box = [x1, y1, x2, y2]   # format of bounding_boxes[i]
    for box in range(0, len(traffic_boxes)):
        if traffic_boxes[box][0] > img_middle and traffic_boxes[box][2] > img_middle :  # bounding box is on the right side of the camera
            light_boxes.append(traffic_boxes[box])
            print (light_boxes[-1])
    y_of_light = 400 # arbitrary value that is used to compare when there is more than one detected traffic light
    if len(light_boxes) < 1:
        print("DEBUG: oh no! there aren't any boxes!") # exit frame and try again
    # we only want to look at one light, so if we detect more than one,
    # we will look at the traffic light that is closest to the top of the pic as
    # that one is likely to be the one we want to look at
    else:
        if len(light_boxes) > 1 :
            for i in range(0, len(light_boxes)) :
                top_y = min(light_boxes[i][1], light_boxes[i][3])
                if top_y < y_of_light :
                    y_of_light = top_y
                    desired_light = i
        else :
            desired_light = 0   # there's only one traffic light detected in the desired region

        # crop image:
        x1 = int(light_boxes[desired_light][0])
        y1 = int(light_boxes[desired_light][1])
        x2 = int(light_boxes[desired_light][2])
        y2 = int(light_boxes[desired_light][3])
        cropped_img = img[y1:y2, x1:x2]
        '''
        color_detected = predict_color(cropped_img)
        print (color_detected)
		file = open('yolo_file.txt','w') # write to a file
		file.write(color_detected)
		file.close()'''
		# file = open('yolo_file.txt','r')
		# color = file.readline()
        # print to file
    # end of Sensor stuff
    gc.collect()

    # print("Class ID: ", current_class_id)
    # print("Score: ", current_score)
    # print("Bounding Box Coordinates: ", current_bb, "\n")

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF


vs.release()
cv2.destroyAllWindows()
