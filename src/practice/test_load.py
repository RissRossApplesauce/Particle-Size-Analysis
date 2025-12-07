import cv2
import numpy as np

"""
ideas:
enhance image:
- edge detection
- contrast

process image:
- line directions
    - find normals
- fourier transform along a path?
- image-based frequency algorithms?

frequencies are:
- directional
    - direction comes from orientation of the nanowires
- local
"""



"""
process:
- image to vector field
    - vector field represents brightness gradients
    - use kernel to calculate?
- vector field to paths
- paths to fourier transforms
- fourier transforms to averages of frequencies

"""


def nothing(x):
    pass



img = cv2.imread('input/10_001_ZC.tif')
img: cv2.typing.MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



windowName = 'window'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, 800, 800)
trackbarName = 'Smoothing'
smooth_scale = 100
cv2.createTrackbar(trackbarName, windowName, 0, smooth_scale, nothing)

while True:
    # smooth out noise
    smoothing = cv2.getTrackbarPos(trackbarName, windowName)
    kernel = smoothing * 2 + 1
    smoothed_img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    
    # create vector field
    sobel_size = 5
    sobelx = cv2.Sobel(smoothed_img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobely = cv2.Sobel(smoothed_img, cv2.CV_64F, 0, 1, ksize=sobel_size)
    vector_field = cv2.merge([sobelx, sobely])
    
    cv2.imshow(windowName, smoothed_img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break