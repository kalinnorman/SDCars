import cv2
import numpy as np

#depth = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 5]])
depth = cv2.imread("C:\\Users\\bryan\\Pictures\\ReddBryan.png", cv2.IMREAD_GRAYSCALE)

depth_scaled = ((depth/np.amax(depth))*255).astype(dtype='uint8')
depth_scaled = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_RAINBOW)  # apply color map for pretty colors

cv2.imshow("scaled", depth_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()