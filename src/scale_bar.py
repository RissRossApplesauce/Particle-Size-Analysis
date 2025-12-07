import cv2
import easyocr
import math
import warnings
from config import settings

warnings.filterwarnings("ignore", category=UserWarning)

reader = easyocr.Reader(["en"], gpu=True)


def scale_bar_width(img):
    width_offset = (
        settings.scale_bar_detection.width_offset
    )  # hardcoded number added to the resulting width
    area_min = settings.scale_bar_detection.min_area_ratio * math.prod(img.shape[:2])
    area_max = settings.scale_bar_detection.max_area_ratio * math.prod(img.shape[:2])

    ret, thresh = cv2.threshold(
        img,
        settings.scale_bar_detection.min_brightness_threshold,
        settings.scale_bar_detection.max_brightness_threshold,
        1,
    )
    contours, h = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        if not area_min < cv2.contourArea(cnt) < area_max:
            continue

        polygon_approximation = cv2.approxPolyDP(
            cnt, 0.01 * cv2.arcLength(cnt, True), True
        )
        if len(polygon_approximation) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            return w + width_offset

    return None


def scale_bar_number(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thresh = cv2.threshold(
        blur,
        settings.scale_bar_detection.min_brightness_threshold,
        settings.scale_bar_detection.max_brightness_threshold,
        1,
    )
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("thresh", 800, 800)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    text = reader.readtext(thresh)
    string = "".join([result[1] for result in text if result[2] > 0.7])
    return float(str("".join([char for char in string if char.isnumeric()])))


def pixel_ratio(img, scale=None):
    nanometers = scale if scale is not None else scale_bar_number(img)
    width_pixels = scale_bar_width(img)
    pixels_per_nm = width_pixels / nanometers
    return pixels_per_nm
