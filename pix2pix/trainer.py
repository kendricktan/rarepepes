import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from pycrayon import CrayonClient
from PIL import Image
from torch.autograd import Variable
from models import Generator, Discriminator
from tqdm import tqdm
from utils import normalize


class Pix2PixTrainer:

    def __init__(self, nic, noc, ngf, ndf, beta=0.5, lamb=100, lr=1e-3, cuda=True, crayon=False):
        """
        Args:
            nic: Number of input channel
            noc: Number of output channels
            ngf: Number of generator filters
            ndf: Number of discriminator filters
            lamb: Weight on L1 term in objective
        """
        self.cuda = cuda
        self.start_epoch = 0

        if crayon:
            self.cc = CrayonClient(hostname="localhost", port=8889)

            try:
                self.logger = self.cc.create_experiment('pix2pix')
            except:
                self.cc.remove_experiment('pix2pix')
                self.logger = self.cc.create_experiment('pix2pix')

        self.gen = self.cudafy(Generator(nic, noc, ngf))
        self.dis = self.cudafy(Discriminator(nic, noc, ndf))

        # Optimizers for generators
        self.gen_optim = self.cudafy(optim.Adam(
            self.gen.parameters(), lr=lr, betas=(beta, 0.999)))

        # Optimizers for discriminators
        self.dis_optim = self.cudafy(optim.Adam(
            self.dis.parameters(), lr=lr, betas=(beta, 0.999)))

        # Loss functions
        self.criterion_bce = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        self.lamb = lamb

    def train(self, loader, c_epoch):
        self.dis.train()
        self.gen.train()
        self.reset_gradients()

        max_idx = len(loader)
        for idx, features in enumerate(tqdm(loader)):
            orig_x = Variable(self.cudafy(features[0]))
            orig_y = Variable(self.cudafy(features[1]))

            """ Discriminator """
            # Train with real
            self.dis.volatile = False
            dis_real = self.dis(torch.cat((orig_x, orig_y), 1))
            real_labels = Variable(self.cudafy(
                torch.ones(dis_real.size())
            ))
            dis_real_loss = self.criterion_bce(
                dis_real, real_labels)

            # Train with fake
            gen_y = self.gen(orig_x)
            dis_fake = self.dis(torch.cat((orig_x, gen_y.detach()), 1))
            fake_labels = Variable(self.cudafy(
                torch.zeros(dis_fake.size())
            ))
            dis_fake_loss = self.criterion_bce(
                dis_fake, fake_labels)

            # Update weights
            dis_loss = dis_real_loss + dis_fake_loss
            dis_loss.backward()

            self.dis_optim.step()
            self.reset_gradients()

            """ Generator """
            self.dis.volatile = True
            dis_real = self.dis(torch.cat((orig_x, gen_y), 1))
            real_labels = Variable(self.cudafy(
                torch.ones(dis_real.size())
            ))
            gen_loss = self.criterion_bce(dis_real, real_labels) + \
                self.lamb * self.criterion_l1(gen_y, orig_y)
            gen_loss.backward()
            self.gen_optim.step()

            # Pycrayon or nah
            if self.crayon:
                self.logger.add_scalar_value('pix2pix_gen_loss', gen_loss.data[0])
                self.logger.add_scalar_value('pix2pix_dis_loss', dis_loss.data[0])

            if idx % 50 == 0:
                tqdm.write('Epoch: {} [{}/{}]\t'
                           'D Loss: {:.4f}\t'
                           'G Loss: {:.4f}'.format(
                               c_epoch, idx, max_idx, dis_loss.data[0], gen_loss.data[0]
                           ))

    def test(self, loader, e):
        self.dis.eval()
        self.gen.eval()

        topilimg = transforms.ToPILImage()

        if not os.path.exists('visualize/'):
            os.makedirs('visualize/')

        idx = random.randint(0, len(loader) - 1)
        _features = loader.dataset[idx]

        orig_x = Variable(self.cudafy(_features[0]))
        orig_y = Variable(self.cudafy(_features[1]))

        orig_x = orig_x.view(1, orig_x.size(0), orig_x.size(1), orig_x.size(2))
        orig_y = orig_y.view(1, orig_y.size(0), orig_y.size(1), orig_x.size(3))

        gen_y = self.gen(orig_x)

        if self.cuda:
            orig_x_np = normalize(orig_x.squeeze().cpu().data, 0, 1)
            orig_y_np = normalize(orig_y.squeeze().cpu().data, 0, 1)
            gen_y_np = normalize(gen_y.squeeze().cpu().data, 0, 1)

        else:
            orig_x_np = normalize(orig_x.squeeze().data, 0, 1)
            orig_y_np = normalize(orig_y.squeeze().data, 0, 1)
            gen_y_np = normalize(gen_y.squeeze().data, 0, 1)

        orig_x_np = topilimg(orig_x_np)
        orig_y_np = topilimg(orig_y_np)
        gen_y_np = topilimg(gen_y_np)

        f, (ax1, ax2, ax3) = plt.subplots(
            3, 1, sharey='row'
        )

        ax1.imshow(orig_x_np)
        ax1.set_title('x')

        ax2.imshow(orig_y_np)
        ax2.set_title('target y')

        ax3.imshow(gen_y_np)
        ax3.set_title('generated y')

        f.savefig('visualize/{}.png'.format(e))

    def save(self, e, filename='rarepepe_weights.tar'):
        torch.save({
            'gen': self.gen.state_dict(),
            'dis': self.dis.state_dict(),
            'epoch': e + 1
        }, 'epoch{}_{}'.format(e, filename))
        print('Saved model state')

    def load(self, filedir):
        if os.path.isfile(filedir):
            checkpoint = torch.load(filedir)

            self.gen.load_state_dict(checkpoint['gen'])
            self.dis.load_state_dict(checkpoint['dis'])
            self.start_epoch = checkpoint['epoch']

            print('Model state loaded')

        else:
            print('Cant find file')

    def reset_gradients(self):
        """
        Helper function to reset gradients
        """
        self.gen.zero_grad()
        self.dis.zero_grad()

    def cudafy(self, t):
        """
        Helper function to cuda-fy our graphs
        """
        if self.cuda:
            if hasattr(t, 'cuda'):
                return t.cuda()
        return t
