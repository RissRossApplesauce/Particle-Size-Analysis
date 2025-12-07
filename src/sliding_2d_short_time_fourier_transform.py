import cv2
import numpy as np
import math
from itertools import count

def get_mask(img):
    painting = False
    erasing = False
    x, y = 0, 0
    size = 100
    
    green = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    green[:] = (0, 255, 0)
    mask = np.zeros_like(img, dtype=np.uint8)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    def mouse_callback(event, mouse_x, mouse_y, flags, param):
        nonlocal painting, erasing, mask, x, y, size
        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False
        
        if event in [cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP]:
            x, y = mouse_x, mouse_y
            if painting:
                cv2.circle(mask, (mouse_x, mouse_y), size, 255, -1)
            if erasing:
                cv2.circle(mask, (mouse_x, mouse_y), size, 0, -1)
        
        if event == cv2.EVENT_MOUSEWHEEL:
            prevsize = size
            if flags < 0:
                size *= 1.2
                if int(size) == prevsize:
                    size = prevsize + 1
            elif flags > 0:
                size *= 0.8
            
            if size < 2: size = 2
            size = int(size)
    
    windowName = 'mask'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1000, 1000)
    cv2.setMouseCallback(windowName, mouse_callback)
    
    while True:
        green_masked = cv2.bitwise_and(green, green, mask=mask)
        img_with_mask = cv2.addWeighted(img_color, 0.6, green_masked, 0.4, 0)
        final_image = np.where(mask[:, :, None] == 255, img_with_mask, img_color)
        cv2.circle(final_image, (x, y), size, (255, 0, 0), 2)
        cv2.imshow(windowName, final_image)
        if cv2.waitKey(5) & 0xFF == ord(' '):
            return mask

def get_gaussian_size(sigma):
    size = int(6 * sigma) # 3 standard deviations in each direction
    if size % 2 == 0:
        size += 1
    
    return size

def gaussian_window(sigma):
    size = get_gaussian_size(sigma)
    
    axis = np.arange(size) - (size // 2)
    xx, yy = np.meshgrid(axis, axis)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    return kernel

def extract_dominant_freq(img, min_freq):
    # FFT
    img = img - np.mean(img)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # find max
    rows, cols = magnitude.shape
    min_freq_adjusted = min_freq * cols
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - cx)**2 + (y - cy)**2 <= min_freq_adjusted**2
    magnitude[mask] = 0
    y_max, x_max = map(float, np.unravel_index(np.argmax(magnitude), magnitude.shape))
    
    # # visualization
    # magnitude_spectrum = 20*np.log(np.abs(f_shift) + 1e-9)
    # magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # magnitude_spectrum_normalized[mask] = 0
    # 
    # magnitude_spectrum_normalized[int(y_max),int(x_max)] = 255
    # 
    # def move_callback(event, x, y, flags, param):
    #     nonlocal magnitude
    #     if event == cv2.EVENT_LBUTTONUP:
    #         print(magnitude[y,x])
    # 
    # windowName = 'magnitude spectrum'
    # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(windowName, 1000, 1000)
    # cv2.setMouseCallback(windowName, move_callback)
    # cv2.imshow(windowName, magnitude_spectrum_normalized)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    
    # process results
    freq_y = (y_max - cy) / img.shape[0]
    freq_x = (x_max - cx) / img.shape[1]
    freq = (freq_y ** 2 + freq_x ** 2) ** 0.5
    # angle = math.atan2(freq_y, freq_x)
    return freq

def sliding_2d_short_time_fourier_transform(
    img: cv2.typing.MatLike, # image to perform transform on, should be grayscale
    sigma, # pixels per standard deviation - used to create a gaussian window at each point
    stride, # number of pixels to skip between points. the output shape will be img.shape // stride (i think)
    min_freq, # smallest allowable frequency (in wires per pixel) to be considered a 'result' from the fourier transform. required because fourier transforms give high values for small frequencies
    mask: cv2.typing.MatLike = None, # mask to control which areas solutions are calculated for, must be the same shape as img
):
    # TODO rename min_freq to something more meaningful
    img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    if mask is None:
        mask = np.ones(img.shape, np.float32)
    
    # sliding window
    height, width = img.shape
    halfstride = stride // 2
    gaussian_size = get_gaussian_size(sigma)
    pad = gaussian_size // 2
    
    # padding around image and mask
    padded_img = np.pad(img, pad_width=pad, mode='constant', constant_values=0)
    padded_mask = np.pad(mask, pad_width=pad, mode='constant', constant_values=0)
    
    # prepare output structure
    out_w = len(range(halfstride + pad, width - halfstride + pad + 1, stride))
    out_h = len(range(halfstride + pad, height - halfstride + pad + 1, stride))
    out = np.zeros((out_h, out_w), np.float32)
    
    for out_x, x in enumerate(range(halfstride + pad, width - halfstride + pad + 1, stride)):
        for out_y, y in enumerate(range(halfstride + pad, height - halfstride + pad + 1, stride)):
            if padded_mask[y,x] == 0:
                continue
            
            # apply gaussian window
            gw = gaussian_window(sigma)
            gw_h, gw_w = gw.shape
            
            # area around x,y, same size as gaussian window
            img_slice = padded_img[y - (gw_h // 2):y + (gw_h // 2) + 1, x - (gw_w // 2):x + (gw_w // 2) + 1]
            img_slice_product = img_slice * gw
            
            freq = extract_dominant_freq(img_slice_product, min_freq)
            
            out[out_y,out_x] = freq
    
    # # for visual analysis, could maybe return these values in the future
    # # note the shape of out has changed from the original [out_y, out_x, 2] with freqs in out[:, :, 0]
    # max_freq = out[:, :, 0].max()
    # 
    # out_hue = np.uint8(255 * out[:, :, 0] / max_freq)
    # 
    # out_img = np.full((out.shape[0], out.shape[1], 3), 255, dtype=np.uint8)
    # out_img[:, :, 0] = out_hue[:, :]
    # 
    # out_rgb = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)
    # 
    # # cv2.imshow('img', img)
    # windowName = 'output'
    # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(windowName, 1000, 1000)
    # cv2.imshow(windowName, out_rgb)
    # cv2.waitKey(0)
    
    # result analysis using IQR method to throw out outliers
    # TODO handle result analysis outside of this function
    data = out.flatten()
    data = data[(data != 0)]
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    cleaned_avg = np.average(clean_data)
    return cleaned_avg

