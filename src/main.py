import cv2
import glob

from scale_bar import pixel_ratio
from sliding_2d_short_time_fourier_transform import get_mask, sliding_2d_short_time_fourier_transform

# features to consider adding:
#   allow users to configure sigma/stride/min_freq
#   show where on the input image the strides will land
#   show how big the gaussian window will turn out
#   after masking, further remove outliers?
#       currently doing basic interquartile range outlier removal in sliding_2d_short_time_fourier_transform
#       outliers could maybe be identified in 2 ways: angle out of phase, significantly different freq


files = glob.glob('input/*')
# TODO figure out reliable way to extract text from image, or give the user an option to input a value if it fails
scales = [20, 10, 20, 10, 10, 10, 10]
references = [0.671,0.623,0.632,0.639,0.627,0.613,0.620]

for file, scale, reference in zip(files, scales, references):
    img = cv2.imread(file, 0)
    mask = get_mask(img)
    # TODO make this less cheaty, it has a hardcoded offset to count the borders around the bar
    pr = pixel_ratio(img, scale) # pixels per nm
    
    # TODO consider adding an expected_wire_size that can be used on an image-by-image basis if necessary, to choose good sigma and min_freq
    
    expected_wire_size = 0.6 # nm / wire
    expected_wires_per_pixel = (1 / expected_wire_size) * (1 / pr)
    expected_pixels_per_wire = 1 / expected_wires_per_pixel
    min_wires_per_pixel = expected_wires_per_pixel * (2 / 3) # goal is to be less any real value but greater than any significant amount of noise
    sigma = int(expected_pixels_per_wire * 4) # (arbitrary) goal is for 1 std dev to cover ~8 wires
    stride = 64
    freq = sliding_2d_short_time_fourier_transform(img, sigma, stride, min_wires_per_pixel, mask) # wires per pixel
    nm_per_wire = (1 / pr) * (1 / freq)
    print(f'{nm_per_wire:.3f} - {reference:.3f}')