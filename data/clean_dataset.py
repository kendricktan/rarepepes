import os
import argparse
import glob

import skimage.color as skcolor
import skimage.io as skio
import skimage.filters as skfilters
import skimage.feature as skfeature
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pepe cleanser')
    parser.add_argument('--img-dir', required=True, type=str)
    parser.add_argument('--out-dir', required=True, type=str)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.img_dir, '*.png')) + \
        glob.glob(os.path.join(args.img_dir, '*.jpg'))

    for i in tqdm(range(len(files))):
        f = files[i]
        filename = f.split('/')[-1]

        # Get image and convert to grayscale
        img = skio.imread(f)
        gray = skcolor.rgb2gray(img)

        # Get mask to get true rare pepes
        thres = skfilters.threshold_otsu(gray)
        mask = gray >= thres

        # Canny
        edges = skfeature.canny(gray)
        edges = edges * 255

        skio.imsave(os.path.join(args.out_dir, '{}'.format(filename)), edges)
