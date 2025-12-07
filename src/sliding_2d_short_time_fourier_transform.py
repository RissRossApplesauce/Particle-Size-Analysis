import cv2
import numpy as np
from config import settings
import math


def get_gaussian_size(sigma):
    size = int(6 * sigma)  # 3 standard deviations in each direction
    if size % 2 == 0:
        size += 1

    return size


def gaussian_window(sigma):
    size = get_gaussian_size(sigma)

    axis = np.arange(size) - (size // 2)
    xx, yy = np.meshgrid(axis, axis)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    return kernel


def overlay_transparent(background, overlay, x, y):
    """
    Overlays a BGRA image onto a BGR background at position (x, y).
    Handles clipping if the overlay extends beyond the background boundaries
    or if x/y are negative.
    """
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    x -= ov_w // 2
    y -= ov_h // 2

    # 1. Calculate the bounding box of the intersection
    #    (The rectangle where the two images actually overlap)

    # Start coordinates (on background)
    bg_x1 = max(x, 0)
    bg_y1 = max(y, 0)

    # End coordinates (on background)
    bg_x2 = min(x + ov_w, bg_w)
    bg_y2 = min(y + ov_h, bg_h)

    # 2. Check if there is any overlap at all
    #    If x1 >= x2 or y1 >= y2, the overlay is completely off-screen
    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background

    # 3. Calculate the corresponding coordinates on the Overlay image
    #    If x was negative (e.g. -10), we skip the first 10 pixels of the overlay
    ov_x1 = bg_x1 - x
    ov_y1 = bg_y1 - y

    #    The width/height of the slice must match the background slice
    ov_x2 = ov_x1 + (bg_x2 - bg_x1)
    ov_y2 = ov_y1 + (bg_y2 - bg_y1)

    # 4. Extract the Slices
    bg_slice = background[bg_y1:bg_y2, bg_x1:bg_x2]
    ov_slice = overlay[ov_y1:ov_y2, ov_x1:ov_x2]

    # 5. Perform the Alpha Blending (Same logic as before)
    alpha = ov_slice[:, :, 3] / 255.0
    overlay_rgb = ov_slice[:, :, :3]

    alpha_factor = alpha[:, :, np.newaxis]

    blended = (overlay_rgb * alpha_factor) + (bg_slice * (1.0 - alpha_factor))

    # 6. Put the blended result back
    output = background.copy()
    output[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)

    return output


def get_mask(img, stride=None, sigma=None):
    painting = False
    erasing = False
    showing_sample = False
    x, y = 0, 0
    size = settings.mask_painter.default_radius

    pointer_color = (255, 0, 0)
    stride_indicator_color = (0, 255, 255)
    paint_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    paint_color[:] = settings.mask_painter.paint_color
    mask = np.zeros_like(img, dtype=np.uint8)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # add stride markers
    if stride is not None:
        halfstride = stride // 2
        for x in range(halfstride, img.shape[1] - halfstride + 1, stride):
            for y in range(halfstride, img.shape[0] - halfstride + 1, stride):
                cv2.circle(img_color, (x, y), 3, stride_indicator_color, -1)

    def mouse_callback(event, mouse_x, mouse_y, flags, param):
        nonlocal painting, erasing, showing_sample, mask, x, y, size
        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            showing_sample = True
        elif event == cv2.EVENT_MBUTTONUP:
            showing_sample = False

        if event in [
            cv2.EVENT_MOUSEMOVE,
            cv2.EVENT_LBUTTONDOWN,
            cv2.EVENT_LBUTTONUP,
            cv2.EVENT_RBUTTONDOWN,
            cv2.EVENT_RBUTTONUP,
        ]:
            x, y = mouse_x, mouse_y
            if painting:
                cv2.circle(mask, (mouse_x, mouse_y), size, 255, -1)
            if erasing:
                cv2.circle(mask, (mouse_x, mouse_y), size, 0, -1)

        if event == cv2.EVENT_MOUSEWHEEL:
            prevsize = size
            if flags < 0:
                size *= settings.mask_painter.scroll_factor
                if int(size) == prevsize:
                    size = prevsize + 1
            elif flags > 0:
                size /= settings.mask_painter.scroll_factor

            if size < 2:
                size = 2
            size = int(size)

    windowName = "mask"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, settings.app.window_size, settings.app.window_size)
    cv2.setMouseCallback(windowName, mouse_callback)

    while True:
        colored_mask = cv2.bitwise_and(paint_color, paint_color, mask=mask)
        op = settings.mask_painter.paint_opacity
        painted_image = cv2.addWeighted(img_color, 1 - op, colored_mask, op, 0)
        painted_image = np.where(mask[:, :, None] == 255, painted_image, img_color)

        # add sample label
        if stride is not None:
            halfstride = stride // 2
            marker_count = 0
            for _x in range(halfstride, img.shape[1] - halfstride + 1, stride):
                for _y in range(halfstride, img.shape[0] - halfstride + 1, stride):
                    if mask[_y, _x] != 0:
                        marker_count += 1
            text = f"{marker_count} samples"
            cv2.putText(
                painted_image,
                text,
                (10, painted_image.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                15,
            )
            cv2.putText(
                painted_image,
                text,
                (10, painted_image.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
            )

        # show size of gaussian sample when middle clicking
        if sigma is not None and showing_sample:
            gw = gaussian_window(sigma)
            gaussian_sample_color = (0, 255, 255)
            gw_color_bgra = np.full((gw.shape[0], gw.shape[1], 4), 0, np.uint8)
            gw_color_bgra[:, :, :3] = gaussian_sample_color
            gw_color_bgra[:, :, 3] = gw * 255  # alpha channel
            painted_image = overlay_transparent(painted_image, gw_color_bgra, x, y)
            cv2.circle(painted_image, (x, y), sigma, gaussian_sample_color, 2)
            cv2.circle(painted_image, (x, y), 2 * sigma, gaussian_sample_color, 2)
            cv2.putText(
                painted_image,
                f"Gaussian Distribution:",
                (x - 340, y - 2 * sigma - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
            )
            cv2.putText(
                painted_image,
                "1s",
                (x + sigma - 40, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
            )
            cv2.putText(
                painted_image,
                "2s",
                (x + 2 * sigma - 40, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
            )
        else:
            cv2.circle(painted_image, (x, y), size, pointer_color, 2)
        cv2.imshow(windowName, painted_image)
        if cv2.waitKey(5) & 0xFF == ord(settings.app.next_key) and mask.any():
            return mask


def extract_dominant_freq(img, min_freq, max_freq):
    # FFT
    img = img - np.mean(img)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # find max
    rows, cols = magnitude.shape
    min_freq_adjusted = min_freq * cols
    max_freq_adjusted = max_freq * cols
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    mask = (dist_sq <= min_freq_adjusted**2) | (max_freq_adjusted**2 <= dist_sq)
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
    freq = (freq_y**2 + freq_x**2) ** 0.5
    # angle = math.atan2(freq_y, freq_x)
    return freq


def sliding_2d_short_time_fourier_transform(
    img: cv2.typing.MatLike,  # image to perform transform on, should be grayscale
    sigma: int = 8,  # pixels per standard deviation - used to create a gaussian window at each point
    stride: int = 64,  # number of pixels to skip between points. the output shape will be img.shape // stride (i think)
    min_freq: float = 0,  # smallest allowable frequency (in wires per pixel) to be considered a 'result' from the fourier transform. required because fourier transforms give high values for small frequencies
    max_freq: float = math.inf,  # similar to min_freq, provides an upper bound on frequencies
    mask: cv2.typing.MatLike = None,  # mask to control which areas solutions are calculated for, must be the same shape as img
):
    img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

    if mask is None:
        mask = np.ones(img.shape, np.float32)

    # sliding window
    height, width = img.shape
    halfstride = stride // 2
    gaussian_size = get_gaussian_size(sigma)
    pad = gaussian_size // 2

    # padding around image and mask
    padded_img = np.pad(img, pad_width=pad, mode="constant", constant_values=0)
    padded_mask = np.pad(mask, pad_width=pad, mode="constant", constant_values=0)

    # prepare output structure
    out_w = len(range(halfstride + pad, width - halfstride + pad + 1, stride))
    out_h = len(range(halfstride + pad, height - halfstride + pad + 1, stride))
    freq_map = np.full((out_h, out_w), np.nan, np.float32)

    for out_x, x in enumerate(
        range(halfstride + pad, width - halfstride + pad + 1, stride)
    ):
        for out_y, y in enumerate(
            range(halfstride + pad, height - halfstride + pad + 1, stride)
        ):
            if padded_mask[y, x] == 0:
                continue

            # apply gaussian window
            gw = gaussian_window(sigma)
            gw_h, gw_w = gw.shape

            # area around x,y, same size as gaussian window
            img_slice = padded_img[
                y - (gw_h // 2) : y + (gw_h // 2) + 1,
                x - (gw_w // 2) : x + (gw_w // 2) + 1,
            ]
            img_slice_product = img_slice * gw

            freq = extract_dominant_freq(img_slice_product, min_freq, max_freq)

            freq_map[out_y, out_x] = freq

    return freq_map


def iqr_filter(data):
    # result analysis using IQR method to throw out outliers
    data = data[~np.isnan(data)]
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return list(clean_data)


def z_score_filter(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    return data[np.abs(z_scores) < settings.data.z_score_filter_threshold]


def visualize_freq_map(freq_map):
    max_freq = freq_map[~np.isnan(freq_map)].max()
    min_freq = freq_map[~np.isnan(freq_map)].min()
    if min_freq == max_freq:
        float_strengths = 177 * (freq_map) / max_freq
    else:
        float_strengths = 100 * (freq_map - min_freq) / (max_freq - min_freq) + 77
    clean_strengths = np.nan_to_num(float_strengths, nan=0)
    rel_strengths = clean_strengths.astype(np.uint8)
    color_img_hsv = np.full(
        (freq_map.shape[0], freq_map.shape[1], 3), 255, dtype=np.uint8
    )
    color_img_hsv[:, :, 2] = rel_strengths
    color_img_hsv[:, :, 0] = 127
    img_bgr = cv2.cvtColor(color_img_hsv, cv2.COLOR_HSV2BGR)
    windowName = "freqs"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 800, 800)
    cv2.imshow(windowName, img_bgr)
    cv2.waitKey(0)
