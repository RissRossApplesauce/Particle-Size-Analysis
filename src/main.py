import cv2
from fourier_analysis import strongest_freq
import glob

files = glob.glob('input/*')
scales = [20, 10, 20, 10, 10, 10, 10]

for file, scale in zip(files, scales):
    img = cv2.imread(file, 0)
    print(f'{file}: {strongest_freq(img, scale):.2f} nm')