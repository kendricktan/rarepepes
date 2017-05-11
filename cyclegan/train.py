import time
import argparse
import torch
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms

from loader import PepeLoader
from trainer import CycleGANTrainer

parser = argparse.ArgumentParser(description='rarepepe trainer')
parser.add_argument('--img-dir', required=True, type=str)
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--cuda', default='true', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--mode', default='train', type=str,
                    help='[train | generate | test]')
args, unknown = parser.parse_known_args()

cuda = 'true' in args.cuda.lower()
train = 'train' in args.mode.lower()

if __name__ == '__main__':
    # in_dim, h_dim, z_dim
    trainer = CycleGANTrainer([3, 8, 32, 64], [32, 8, 3], lr=args.lr, cuda=cuda)

    if args.resume:
        trainer.load(args.resume)

    dataset = PepeLoader(
        args.img_dir,
        transform=transforms.Compose([
            transforms.Scale(300),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ]),
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
