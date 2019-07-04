from PIL import Image
import numpy as np
from os import listdir
from skimage.io import imread
from skimage.transform import resize
from skimage.io import imsave
import os
import copy
import math

from os.path import isfile, isdir

root_dir = os.getcwd()

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 480

# Edit these
DATA_DIR = os.path.join(root_dir,'data')
OUTPUT_DIR = "mapped/"

PATHS = [os.path.join(DATA_DIR, f) for f in listdir(DATA_DIR) if isfile(os.path.join(DATA_DIR, f))]
single_image_path = PATHS[0]


def load_colors(use_mm = False):
    rgbs = {}
    with open('colors.txt') as f:
        for line in f.read().splitlines():
            r,g,b,mm,dbz = line.split(',')
            rainfall = 255 - 255 * ((float(mm) / 200.0) if use_mm else (float(dbz) / 60.0))
            rgbs[tuple(int(x, base=16) for x in (r,g,b))] = rainfall
    return rgbs

def calculate_boundary(width, height, threshold):
    boundary = np.zeros((width, height), dtype=bool)
    center_x, center_y = (width / 2, height / 2)
    for x in range(width):
        for y in range(height):
            if math.hypot(center_x - x, center_y - y) > threshold:
                boundary[x, y] = True
    return boundary

ALLOWED = load_colors()
BOUNDARY = calculate_boundary(IMAGE_WIDTH, IMAGE_HEIGHT, 250.0)

def map_color(color):
    key = (color[0], color[1], color[2])
    return ALLOWED.get(key, 255)

def convert_image(path):
    def reds(c):
        return (0, 0, 0) if c[0] == 255 and c[1] == 0 and c[2] == 0 else c

    image = imread(path)[:, :, :3]
    basename = os.path.basename(path)
    image[BOUNDARY] = np.apply_along_axis(reds, 1, image[BOUNDARY])
    new_array = np.apply_along_axis(map_color, 2, image)
    imsave(os.path.join(OUTPUT_DIR, basename), new_array)
    return image


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in PATHS:
        convert_image(path)


if __name__ == "__main__":
    main()