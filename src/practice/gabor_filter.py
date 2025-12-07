"""
lambda = wavelength - vary this
theta = orientation (rotation of wavelet thingy) - vary this
sigma = standard deviation of gaussian - pick good constant
gamma = aspect ratio (leave 1?)
psi = phase offset (leave 0)


create gabor filter bank
convolve image with each filter in the bank

"""

import numpy as np
import cv2
import math

def nothing(x):
    pass

mouse_x, mouse_y = 0, 0

def move_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


img = cv2.imread(r'input\16_001_ZC.tif', 0)
# img = cv2.GaussianBlur(img, (17, 17), 0)


windowName = 'window'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, 800, 800)
cv2.setMouseCallback(windowName, move_callback)

gabor_kernels = [
    cv2.getGaborKernel(ksize=(131,131), sigma=3, theta=theta, lambd=wavelength, gamma=0)
    for theta in np.arange(0, np.pi, np.pi / 16)
    for wavelength in np.arange(100, 300, 10)
]

labels = [
    f'Theta: {theta} - Lambda: {wavelength}'
    for theta in np.arange(0, np.pi, np.pi / 16)
    for wavelength in np.arange(100, 300, 10)
]

print(len(gabor_kernels))

gabor_idx = 0
while True:
    # frequency = freq[int(mouse_y),int(mouse_x)]
    # wavelength = 1 / frequency
    
    display_img = cv2.filter2D(img, cv2.CV_64F, gabor_kernels[gabor_idx])
    display_img = cv2.convertScaleAbs(display_img, alpha=1.0)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    text = f'{gabor_idx} - {labels[gabor_idx]}'
    cv2.putText(display_img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    # rads = math.radians(orientation)
    # dx, dy = math.cos(rads) * wavelength, -math.sin(rads) * wavelength
    # cv2.line(display_img, (mouse_x, mouse_y), (int(mouse_x + dx), int(mouse_y + dy)), (255, 0, 0), 4)
    
    cv2.imshow(windowName, display_img)
    if (key := cv2.waitKey(5) & 0xFF):
        if key == ord('q'):
            break
        if key == ord('s'):
            gabor_idx += 1
            if gabor_idx >= len(gabor_kernels):
                gabor_idx = 0
        if key == ord('a'):
            gabor_idx -= 1
            if gabor_idx < 0:
                gabor_idx = len(gabor_kernels) - 1