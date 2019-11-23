# USAGE
# sudo MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 yolo.py
# OPTIONAL PARAMETERS
# -c/--confidence (.0-1.0) (detected objects with a confidence higher than this will be used)

# import the necessary packages
#from CarControl import CarControl #steer, drive
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
import serial
import time
import cv2
import os
import gc

class Yolo:

	def __init__(self):
		self.ap = argparse.ArgumentParser()
		self.ap.add_argument("-c", "--confidence", type=float, default=0.5,
		    help="minimum probability to filter weak detections")
		self.args = vars(self.ap.parse_args())

		(self.W, self.H) = (None, None)

		# Implement YOLOv3MXNet
		self.net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

		# Set device to GPU
		self.device=mx.gpu()
		# device=mx.cpu()

		self.net.collect_params().reset_ctx(self.device)

		signal(SIGINT, self.handler)
		print('Running. Press CTRL-C to exit')

		os.system('MXNET_CUDNN_AUTOTUNE_DEFAULT=0')


	"""Transforms for YOLO series."""
	def transform_test(self, imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
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

	def load_test(self, filenames, short=416):
	    if not isinstance(filenames, list):
	        filenames = [filenames]
	    imgs = [self.letterbox_image(f, short) for f in filenames]
	    return self.transform_test(imgs)


	# this function is from yolo3.utils.letterbox_image
	def letterbox_image(self, image, size=416):
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
	def handler(self, signal_received, frame):
	    vs.release()
	    cv2.destroyAllWindows()
	    print('CTRL-C detected. Exiting gracefully')
	    exit(0)

	def main_yolo(self, vs) :

		current_bb = [0, 0, 0, 0]
		frame = vs
		print("DEBUG: I'm in YOLO!")

		# if the frame dimensions are empty, grab them
		if self.W is None or self.H is None:
		    (self.H, self.W) = frame.shape[:2]

		# from gluoncv import data
		yolo_image = Image.fromarray(frame, 'RGB')
		x, img = self.load_test(yolo_image, short=416)

		class_IDs, scores, bounding_boxs = self.net(x.copyto(self.device))

		# The next two lines draw boxes around detected objects
		ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=self.net.classes)
		plt.show()

		# Convert to numpy arrays, then to lists
		class_IDs = class_IDs.asnumpy().tolist()
		scores = scores.asnumpy().tolist()
		bounding_boxs = bounding_boxs.asnumpy()
		traffic_boxes = []
		# iterate through detected objects
		for i in range(len(class_IDs[0])):
			if ((scores[0][i])[0]) > self.args["confidence"]:
				current_class_id = self.net.classes[int((class_IDs[0][i])[0])]
				current_score = (scores[0][i])[0]
				current_bb = bounding_boxs[0][i-1]
				if current_class_id == 'traffic light':
					traffic_boxes.append(current_bb)
		gc.collect()

		# print("Class ID: ", current_class_id)
		# print("Score: ", current_score)
		# print("Bounding Box Coordinates: ", current_bb, "\n")

		cv2.imshow("Camera Feed", frame)
		key = cv2.waitKey(1) & 0xFF

		# vs.release()
		cv2.destroyAllWindows()

		return traffic_boxes, img
