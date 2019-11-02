import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
count = 0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        count += 1
        if count == 12:
            cv2.imwrite("Depth.jpg",depth_colormap)


finally:

    # Stop streaming
    pipeline.stop()




# import cv2
# import numpy as np 
# import pyrealsense2 as rs
# from matplotlib import pyplot as plt


# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# pipeline.start(config)

# path = "C:/Users/benjj/Documents/College/Fall2019/ECEN522/Code/SDCars/class_code/depth_image.jpg"
# depthImage = cv2.imread(path)

# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)


# plt.imshow(depth_colormap) 
# plt.show()
