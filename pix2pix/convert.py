import utils
import os
import time

import torch
import torch.utils.data
import torchvision.transforms as transforms

import skimage.color as skcolor
import skimage.io as skio
import skimage.filters as skfilters
import skimage.feature as skfeature

import numpy as np

from tqdm import tqdm
from utils import normalize, tensor2im
from PIL import Image
from torch.autograd import Variable


def convert_image(img, model, transformers):
    # Apply transformations and normalizations
    a_img = transformers(img)
    a_img = a_img.view(1, a_img.size(0), a_img.size(1), a_img.size(2))
    a_img = normalize(a_img, -1, 1)

    b_img = torch.randn(a_img.size())

    # Get fake_y (generated output)
    model.set_input({
        'A': a_img,
        'B': b_img
    })
    model.test()
    visuals = model.get_current_visuals()

    return Image.fromarray(visuals['fake_B'])
