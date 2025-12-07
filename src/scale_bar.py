import cv2
import easyocr
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

reader = easyocr.Reader(['en'], gpu=True)


def scale_bar_width(img):
    width_offset = 10 # hardcoded number added to the resulting width
    
    ret, thresh = cv2.threshold(img, 250, 255, 1)
    contours, h = cv2.findContours(thresh, 1, 2)

    img_area = math.prod(img.shape[:2])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.001 * img_area or area > 0.01 * img_area:
            continue
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            return w + width_offset
    
    return None


def scale_bar_number(img):
    img = cv2.GaussianBlur(img, (17, 17), 0)
    ret, thresh = cv2.threshold(img, 210, 255, 1)
    text = reader.readtext(thresh)
    string = ''.join([result[1] for result in text if result[2] > 0.7])
    return float(str(''.join([char for char in string if char.isnumeric()])))
    

def pixel_ratio(img, scale = None):
    if scale == None:
        val = scale_bar_number(img)
    else:
        val = scale
    width = scale_bar_width(img)

    pixels_per_nm = width / val

    return pixels_per_nm