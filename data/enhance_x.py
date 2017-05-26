import os
import argparse
import glob
import sys

import skimage.filters as skfilters
import skimage.color as skcolor
import skimage.feature as skfeature
import skimage.io as skio
import skimage.util as skut
import skimage.morphology as skmo
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pepe data X enhancer')
    parser.add_argument('--img-dir', required=True, type=str)
    args = parser.parse_args()

    # Images
    files = glob.glob(os.path.join(args.img_dir, '*.png')) + \
        glob.glob(os.path.join(args.img_dir, '*.jpg'))

    # Invert and dilate
    for i in tqdm(range(len(files))):
        f = files[i]

        img = skio.imread(f)
        gray = skcolor.rgb2gray(img)

        # Canny
        edges = skfeature.canny(gray, sigma=0.69)
        dilated = skmo.dilation(edges, skmo.square(3))
        eroded = skmo.erosion(dilated, skmo.square(2))

        skio.imshow(eroded)
        plt.show()

