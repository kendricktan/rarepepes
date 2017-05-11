import os
import argparse
import glob
import sys

import skimage.color as skcolor
import skimage.io as skio
import skimage.filters as skfilters
import skimage.feature as skfeature
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

# Fucking imperative programming :-(
N_KEY = False
Y_KEY = False


def plt_keypress(e):
    global N_KEY, Y_KEY
    sys.stdout.flush()

    if e.key == 'y':
        Y_KEY = True
    elif e.key == 'n':
        N_KEY = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pepe data cleanser')
    parser.add_argument('--img-dir', required=True, type=str)
    parser.add_argument('--out-dir', required=True, type=str)
    args = parser.parse_args()

    print('Welcome to pepe cleanser, press "y" to add displayed'
          'image to database and "n" to skip displayed image')

    # Images
    files = glob.glob(os.path.join(args.img_dir, '*.png')) + \
        glob.glob(os.path.join(args.img_dir, '*.jpg'))

    # Contains canny applied to filtered images
    canny_out_dir = os.path.join(args.out_dir, 'x')
    if not os.path.exists(canny_out_dir):
        os.makedirs(canny_out_dir)

    # Contains filtered images
    raw_out_dir = os.path.join(args.out_dir, 'y')
    if not os.path.exists(raw_out_dir):
        os.makedirs(raw_out_dir)

    # Interactive mode
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.canvas.mpl_connect('key_press_event', plt_keypress)

    img_idx = 0
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

        # Clear axis
        plt.cla()

        # Plot and compare
        ax1.clear()
        ax1.imshow(img)
        ax1.set_title('original')

        ax2.clear()
        ax2.imshow(edges)
        ax2.set_title('edges')

        # Update plot
        plt.draw()

        while not Y_KEY and not N_KEY:
            plt.waitforbuttonpress(0)

        # If user hits 'Y' then save image
        if Y_KEY:
            skio.imsave(os.path.join(
                raw_out_dir, '{}.png'.format(img_idx)), img)
            skio.imsave(os.path.join(canny_out_dir,
                                     '{}.png'.format(img_idx)), edges)
            tqdm.write('Saved {}.png'.format(img_idx))
            img_idx += 1

        else:
            tqdm.write('Skipped image')

        Y_KEY = False
        N_KEY = False
