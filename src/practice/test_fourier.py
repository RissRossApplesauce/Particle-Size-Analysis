import cv2
import numpy as np
import math

def nothing(x):
    pass

mouse_x, mouse_y = 0, 0
zoom = 50
invert = False
disable = False

def move_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, zoom, invert, disable
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    if event == cv2.EVENT_MOUSEWHEEL:
        prevzoom = zoom
        if flags > 0:
            zoom *= 1.2
            if int(zoom) == prevzoom:
                zoom = prevzoom + 1
        elif flags < 0:
            zoom *= 0.8
        
        if zoom < 2: zoom = 2
        zoom = int(zoom)
    if event == cv2.EVENT_LBUTTONUP:
        invert = not invert
    if event == cv2.EVENT_RBUTTONUP:
        disable = not disable
        

img = cv2.imread(r'input\15_001_ZC.tif', 0)
img = cv2.GaussianBlur(img, (11, 11), 0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
normalized_magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

windowName = 'window'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, 1000, 1000)
cv2.setMouseCallback(windowName, move_callback)

windowName2 = 'window2'
cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName2, 1000, 1000)

while True:
    shape = img.shape[:2]
    
    # determine point with strongest frequency
    magnitude_copy = np.copy(normalized_magnitude_spectrum)
    rows, cols = magnitude_copy.shape
    crow, ccol = rows // 2, cols // 2
    # adjust brightness based on distance to center
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    radius = np.log(np.sqrt(x**2 + y**2))
    radius[crow, ccol] = 1
    whitened = normalized_magnitude_spectrum * radius
    whitened = cv2.normalize(whitened, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # mask center and work towards smallest frequency
    radius = 25
    cv2.circle(whitened, (crow, ccol), radius, 0, thickness=-1) # mask center
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(whitened)
    freq = ((maxLoc[0] - shape[0] / 2) ** 2 + (maxLoc[1] - shape[1] / 2) ** 2) ** 0.5 # cycles per IMAGE
    wavelength = img.shape[0] / freq
    
    
    # display image
    display_img = np.copy(whitened)
    relative_x, relative_y = mouse_x - shape[0] / 2, mouse_y - shape[1] / 2 # might have indices swapped (will break for non-square images)
    theta = math.degrees(-math.atan2(relative_y, relative_x))
    distance = (relative_x ** 2 + relative_y ** 2) ** 0.5
    text = f'Magnitude: {magnitude_spectrum[mouse_y,mouse_x]:.2f} - Theta: {theta:.2f} - Distance: {distance:.2f}'
    cv2.putText(display_img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    
    # masking based on mouse position, with inverse dft to show what it would look like based on the mask
    h, w = img.shape[:2]
    if not disable:
        mask = np.zeros((h, w), dtype=np.uint8) if not invert else np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(mask, (mouse_x, mouse_y), zoom, 255 if not invert else 0, thickness=-1)
        cv2.circle(display_img, (mouse_x, mouse_y), zoom, 255 if not invert else 0, thickness=2)
        
        inverted_mouse_x, inverted_mouse_y = w - mouse_x, h - mouse_y
        cv2.circle(mask, (inverted_mouse_x, inverted_mouse_y), zoom, 255 if not invert else 0, thickness=-1)
        cv2.circle(display_img, (inverted_mouse_x, inverted_mouse_y), zoom, 255 if not invert else 0, thickness=2)
    else:
        mask = np.ones((h, w), dtype=np.uint8) * 255
    masked_dft_shifted = cv2.bitwise_and(dft_shift, dft_shift, mask=mask)
    masked_dft = np.fft.ifftshift(masked_dft_shifted)
    idft = cv2.idft(masked_dft)
    planes = cv2.split(idft)
    magnitude_image = cv2.magnitude(planes[0], planes[1])
    final_image = cv2.normalize(magnitude_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    
    cv2.imshow(windowName, display_img)
    cv2.imshow(windowName2, final_image)

    if (key := cv2.waitKey(5) & 0xFF):
        if key == ord('q'):
            break