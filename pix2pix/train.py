import matplotlib as mpl
mpl.use('Agg')

import os
import time
import argparse
import torch
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms

from tqdm import tqdm
from options import TrainOptions
from loader import PepeLoader
from models import Pix2PixModel

# CUDA_VISIBLE_DEVICES
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parse options
opt = TrainOptions().parse()

if __name__ == '__main__':
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
        train=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, pin_memory=True
    )

    total_steps = 0

    for e in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        dataset_size = len(dataset)
        for i, data in enumerate(tqdm(dataloader)):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (e - 1)
            model.set_input({
                'A': data[0],
                'B': data[1]
            })
            model.optimize_parameters()

        model.save('latest')
        model.save(e)

        if e > opt.niter:
            model.update_learning_rate()
