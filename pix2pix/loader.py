import os
import glob
import h5py
import random
import PIL.ImageOps
import torch
import torch.utils.data as data
import numpy as np
import skimage.io as skio
import skimage.transform as sktrans
import torchvision.transforms as transforms

from PIL import Image
from utils import normalize


def random_crop(img_size, s):
    w, h = img_size
    th, tw = s

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    return (x1, y1, x1 + tw, y1 + th)


class PepeLoader(data.Dataset):
    def __init__(self, imgdir, invert_x=True, random_seed=42, transform=None, train=True):
        self.x_dir = sorted(glob.glob(os.path.join(
            imgdir, os.path.join('x', '*.png'))))
        self.y_dir = sorted(glob.glob(os.path.join(
            imgdir, os.path.join('y', '*.png'))))

        # Invert x (black to white and vice-versa)
        # Because script in data/clean_dataset.py wasnt inverted
        self.invert_x = invert_x
        self.transform = transform
        self.train = train

        self._length = len(self.x_dir)
        self.train_length = int(self._length * 0.95)
        self.test_length = self._length - self.train_length

        np.random.seed(random_seed)
        self.idxes = np.arange(self._length)
        np.random.shuffle(self.idxes)

        # Scale image
        self.scale = transforms.Scale(300)

    def __getitem__(self, index):
        if not self.train:
            index += self.train_length

        idx = self.idxes[index]

        # Open file as Pillow Image
        x_img = Image.open(self.x_dir[idx])
        y_img = Image.open(self.y_dir[idx])

        if self.invert_x:
            x_img = PIL.ImageOps.invert(x_img)

        x_img = self.scale(x_img.convert('RGB'))
        y_img = self.scale(y_img.convert('RGB'))

        # Randomly crop image
        crop_coord = random_crop(x_img.size, (256, 256))

        x_img = x_img.crop(crop_coord)
        y_img = y_img.crop(crop_coord)

        # Randomly flip
        if random.random() < 0.5:
            x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
            y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            x_img = self.transform(x_img)
            y_img = self.transform(y_img)

        x_img = normalize(x_img, -1, 1)
        y_img = normalize(y_img, -1, 1)

        return x_img, y_img

    def __len__(self):
        if self.train:
            return self.train_length
        return self.test_length
