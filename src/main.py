import cv2
import glob
import numpy as np

from scale_bar import pixel_ratio
from sliding_2d_short_time_fourier_transform import (
    get_mask,
    iqr_filter,
    sliding_2d_short_time_fourier_transform,
    visualize_freq_map,
    z_score_filter,
)
from config import settings

# features to consider adding:
#   throw out values that are not significant enough? right now we produce an output even if there was no true peak
#   since there's an upper bound on accepted frequencies, can the range of the fourier transform be reduced to that upper bound for efficiency?
#   show how big the gaussian window will turn out
#   after masking, further remove outliers?
#       currently doing basic interquartile range outlier removal in sliding_2d_short_time_fourier_transform
#       outliers could maybe be identified in 2 ways: angle out of phase, significantly different freq


files = glob.glob("input/*")
# TODO figure out reliable way to extract text from image, or give the user an option to input a value if it fails
scales = [20, 10, 20, 10, 10, 10, 10]
references = [0.671, 0.606, 0.632, 0.639, 0.627, 0.613, 0.620]

for file, scale, reference in zip(files, scales, references):
    img = cv2.imread(file, 0)

    # determine transform settings
    pixels_per_nm = pixel_ratio(img, scale)
    expected_wires_per_pixel = (1 / settings.data.expected_wire_width) * (
        1 / pixels_per_nm
    )
    stride = img.shape[1] // settings.data.samples_across
    sigma = int(settings.data.wires_per_std_dev / expected_wires_per_pixel / 2)

    mask = get_mask(img, stride, sigma)
    freq_map = sliding_2d_short_time_fourier_transform(
        img,
        sigma,
        stride,
        expected_wires_per_pixel * settings.data.smallest_wire_ratio,
        expected_wires_per_pixel * settings.data.largest_wire_ratio,
        mask,
    )

    # extract results
    inliers = z_score_filter(freq_map)
    iqr_avg = sum(inliers) / len(inliers)
    nm_per_wire = (1 / pixels_per_nm) * (1 / iqr_avg)
    print(f"{nm_per_wire:.3f} - {reference:.3f}")

    visualize_freq_map(np.where(np.isin(freq_map, inliers), freq_map, np.nan))
