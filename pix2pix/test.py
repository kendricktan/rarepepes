import matplotlib as mpl
mpl.use('Agg')

import utils
import os
import time
import argparse
import torch
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tqdm import tqdm
from options import TestOptions
from loader import PepeLoader
from models import Pix2PixModel
from visualizer import Visualizer

# CUDA_VISIBLE_DEVICES
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parse options
opt = TestOptions().parse()


if __name__ == '__main__':
    # pix2pix model
    model = Pix2PixModel()
    model.initialize(opt)

    dataset = PepeLoader(
        opt.dataroot,
        transform=transforms.Compose(
            [transforms.Scale(opt.loadSize),
             transforms.RandomCrop(opt.fineSize),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))
             ]
        ),
        train=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, pin_memory=True
    )

    total_steps = 0

    for idx, data in enumerate(tqdm(dataloader)):
        if idx > opt.how_many:
            break

        model.set_input({
            'A': data[0],
            'B': data[1]
        })
        model.test()
        visuals = model.get_current_visuals()

        utils.mkdir('results')

        f, (ax1, ax2, ax3) = plt.subplots(
            3, 1, sharey='row'
        )

        ax1.imshow(visuals['real_A'])
        ax1.set_title('real A')

        ax2.imshow(visuals['fake_B'])
        ax2.set_title('fake B')

        ax3.imshow(visuals['real_B'])
        ax3.set_title('real B')

        f.savefig('results/{}.png'.format(int(time.time())))
