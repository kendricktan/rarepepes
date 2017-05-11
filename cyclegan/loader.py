import os
import glob
import h5py
import torch
import torch.utils.data as data
import numpy as np
import skimage.io as skio
import skimage.transform as sktrans

from PIL import Image
from utils import normalize


class PepeLoader(data.Dataset):
    def __init__(self, imgdir, random_seed=42, transform=None, train=True):
        self.x_dir = sorted(glob.glob(os.path.join(imgdir, os.path.join('x', '*.png'))))
        self.y_dir = sorted(glob.glob(os.path.join(imgdir, os.path.join('y', '*.png'))))

        self.transform = transform
        self.train = train

        self._length = len(self.x_dir)
        self.train_length = int(self._length * 0.95)
        self.test_length = self._length - self.train_length

        np.random.seed(random_seed)
        self.idxes = np.arange(self._length)
        np.random.shuffle(self.idxes)

    def __getitem__(self, index):
        if not self.train:
            index += self.train_length

        idx = self.idxes[index]

        # Open file as Pillow Image
        x_img = Image.open(self.x_dir[idx])
        y_img = Image.open(self.y_dir[idx])

        x_img = x_img.convert('RGB')
        y_img = y_img.convert('RGB')

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
