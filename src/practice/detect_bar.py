import cv2
import numpy as np
import math

img = cv2.imread(r'input\12_003_Scan.tif', 0)
ret, thresh = cv2.threshold(img, 250, 255, 1)
contours, h = cv2.findContours(thresh, 1, 2)

windowName = 'window'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, 1000, 1000)

img_area = math.prod(img.shape[:2])

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 0.001 * img_area or area > 0.01 * img_area:
        continue
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        print(w, h)
        print(w + 10)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [cnt], -1, (0, 255, 0), 3)
        cv2.imshow(windowName, img_color)
        cv2.waitKey(0)