import time
import argparse
import torch
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms

from loader import PepeLoader
from trainer import Pix2PixTrainer

parser = argparse.ArgumentParser(description='rarepepe trainer')
parser.add_argument('--img-dir', required=True, type=str)
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--beta', default=0.5, type=float)
parser.add_argument('--lamb', default=100, type=float)
parser.add_argument('--cuda', default='true', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--crayon', default='', type=str)
parser.add_argument('--mode', default='train', type=str,
                    help='[train | generate | test]')
args, unknown = parser.parse_known_args()

cuda = 'true' in args.cuda.lower()
train = 'train' in args.mode.lower()
crayon = 'true' in args.crayon.lower()

if __name__ == '__main__':
    # in_dim, h_dim, z_dim
    trainer = Pix2PixTrainer(3, 3, 64, 64, beta=args.beta, lamb=args.lamb, lr=args.lr, cuda=cuda, crayon=crayon)

    if args.resume:
        trainer.load(args.resume)

    dataset = PepeLoader(
        args.img_dir,
        transform=transforms.ToTensor(),
        train=train
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, pin_memory=cuda
    )

    if train:
        for e in range(trainer.start_epoch, args.epoch):
            trainer.train(dataloader, e)
            trainer.save(e)
            trainer.test(dataloader, e)

    else:
        trainer.test(dataloader, int(time.time()))
