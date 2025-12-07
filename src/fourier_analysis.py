import cv2
import numpy as np
import math
from scale_bar import pixel_ratio

def strongest_freq(img, scale = None):
    img = cv2.GaussianBlur(img, (11, 11), 0)
    shape = img.shape
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    magnitude_copy = np.copy(magnitude_spectrum)
    rows, cols = magnitude_copy.shape
    crow, ccol = rows // 2, cols // 2
    radius = 40
    magnitude_copy[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(magnitude_copy)
    freq = ((maxLoc[0] - shape[0] / 2) ** 2 + (maxLoc[1] - shape[1] / 2) ** 2) ** 0.5 # cycles per IMAGE
    wavelength = img.shape[0] / freq # pixels per cycle
    
    pixels_per_nm = pixel_ratio(img, scale)

    return wavelength / pixels_per_nm