import cv2

folder = 'actual_images'
filenames = ['green_bgr_1.png', 'green_bgr_2.png', 'yellow_bgr_1.png', 'red_bgr_1.png', 'red_bgr_2.png']

for filename in filenames:
    img = cv2.imread(folder + '\\' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, img)
