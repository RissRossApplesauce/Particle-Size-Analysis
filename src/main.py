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
    pr = pixel_ratio(img, scale) # pixels per nm
    
    # TODO determine if sigma should just be dependent on min_freq
    # ^rationale: if min_freq is always chosen based on expected wire size ((pixels per wire)^-1 * 0.5) and sigma is always chosen based on expected wire size (pixels per wire * ~4), they're redundant
    sigma = 128 # average wire is about 35 pixels across - this will fit about 16 wires within 2 standard deviations with seems to work well
    stride = 64
    min_freq = 0.015 # with wires being about 35 pixels across, their 'wires per pixel' frequency is around 0.03. divided by 2 to get this number
    # TODO rename min_freq to something more meaningful
    freq = sliding_2d_short_time_fourier_transform(img, sigma, stride, min_freq, mask) # wires per pixel
    print(freq)
    nm_per_wire = (1 / pr) * (1 / freq)
    print(f'{nm_per_wire:.3f} - {reference:.3f}')