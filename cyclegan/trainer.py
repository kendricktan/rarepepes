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
from models import GeneratorCNN, DiscriminatorCNN
from tqdm import tqdm
from utils import normalize


class CycleGANTrainer:

    def __init__(self, conv_dims, convt_dims, lr=1e-3, cuda=True, crayon=True):
        self.cuda = cuda
        self.start_epoch = 0

        self.crayon = crayon
        if crayon:
            self.cc = CrayonClient(hostname="localhost", port=8889)

            try:
                self.logger = self.cc.create_experiment('rarepepe_cyclegan')
            except:
                self.cc.remove_experiment('rarepepe_cyclegan')
                self.logger = self.cc.create_experiment('rarepepe_cyclegan')

        # F(x) -> y
        self.f_xy = self.cudafy(GeneratorCNN(conv_dims, convt_dims))

        # G(y) -> x
        self.g_yx = self.cudafy(GeneratorCNN(conv_dims, convt_dims))

        # D(x) -> real/fake
        self.dis_x = self.cudafy(
            DiscriminatorCNN(conv_dims + convt_dims)
        )

        # D(y) -> real/fake
        self.dis_y = self.cudafy(
            DiscriminatorCNN(conv_dims + convt_dims)
        )

        # Optimizers for domain gens
        # e.g. F(x) and G(y)
        gen_params = list(self.f_xy.parameters()) + \
            list(self.g_yx.parameters())
        self.gen_optim = self.cudafy(optim.Adam(gen_params, lr=lr))

        # Optimizers for discriminators
        # e.g. D(x) and D(y)
        dis_params = list(self.dis_x.parameters()) + \
            list(self.dis_y.parameters())
        self.dis_optim = self.cudafy(optim.Adam(dis_params, lr=lr))

        # Loss functions
        self.gen_loss = nn.MSELoss()

    def train(self, loader, c_epoch):
        self.f_xy.train()
        self.g_yx.train()
        self.dis_x.train()
        self.dis_y.train()
        self.reset_gradients()

        max_idx = len(loader)
        for idx, features in enumerate(tqdm(loader)):
            orig_x = Variable(self.cudafy(features[0]))
            orig_y = Variable(self.cudafy(features[1]))

            """ Discriminator """
            m_xy = self.f_xy(orig_x).detach()
            m_yx = self.g_yx(orig_y).detach()

            loss_dis_x_real = 0.5 * torch.mean((self.dis_x(orig_x) - 1) ** 2)
            loss_dis_x_fake = 0.5 * torch.mean((self.dis_x(m_yx)) ** 2)

            loss_dis_y_real = 0.5 * torch.mean((self.dis_y(orig_y) - 1) ** 2)
            loss_dis_y_fake = 0.5 * torch.mean((self.dis_y(m_xy)) ** 2)

            loss_dis = loss_dis_x_real + loss_dis_x_fake + loss_dis_y_real + loss_dis_y_fake
            loss_dis.backward()
            self.dis_optim.step()
            self.reset_gradients()

            """ Generator """
            m_xy = self.f_xy(orig_x)
            m_yx = self.g_yx(orig_y)

            m_xyx = self.g_yx(m_xy)
            m_yxy = self.f_xy(m_yx)

            loss_const_xy = self.gen_loss(m_xyx, orig_x)
            loss_const_yx = self.gen_loss(m_yxy, orig_y)

            loss_dis_x = 0.5 * torch.mean((self.dis_x(m_yx) - 1) ** 2)
            loss_dis_y = 0.5 * torch.mean((self.dis_y(m_xy) - 1) ** 2)

            loss_gen = loss_const_xy + loss_const_yx + loss_dis_x + loss_dis_y
            loss_gen.backward()
            self.gen_optim.step()
            self.reset_gradients()

            # Pycrayon or nah
            if self.crayon:
                self.logger.add_scalar_value('rarepepe_gen_loss', loss_gen.data[0])
                self.logger.add_scalar_value('rarepepe_dis_loss', loss_dis.data[0])

            if idx % 50 == 0:
                tqdm.write(
                    "Epoch: {} [{}/{}]\t"
                    "G Loss: {:.4f}\t"
                    "D Loss: {:.4f}".format(
                        c_epoch, idx, max_idx,
                        loss_gen.data[0], loss_dis.data[0]
                    )
                )

    def test(self, loader, e):
        self.f_xy.eval()
        self.g_yx.eval()
        self.dis_x.eval()
        self.dis_y.eval()

        topilimg = transforms.ToPILImage()

        if not os.path.exists('visualize/'):
            os.makedirs('visualize/')

        idx = random.randint(0, len(loader) - 1)
        _features = loader.dataset[idx]

        orig_x = Variable(self.cudafy(_features[0]))
        orig_y = Variable(self.cudafy(_features[1]))

        orig_x = orig_x.view(1, orig_x.size(0), orig_x.size(1), orig_x.size(2))
        orig_y = orig_y.view(1, orig_y.size(0), orig_y.size(1), orig_x.size(3))

        mapped_x = self.g_yx(orig_y)
        mapped_y = self.f_xy(orig_x)

        if self.cuda:
            orig_x_np = normalize(orig_x.squeeze().cpu().data, 0, 1)
            orig_y_np = normalize(orig_y.squeeze().cpu().data, 0, 1)
            mapped_x_np = normalize(mapped_x.squeeze().cpu().data, 0, 1)
            mapped_y_np = normalize(mapped_y.squeeze().cpu().data, 0, 1)

        else:
            orig_x_np = normalize(orig_x.squeeze().data, 0, 1)
            orig_y_np = normalize(orig_y.squeeze().data, 0, 1)
            mapped_x_np = normalize(mapped_x.squeeze().data, 0, 1)
            mapped_y_np = normalize(mapped_y.squeeze().data, 0, 1)

        orig_x_np = topilimg(orig_x_np)
        orig_y_np = topilimg(orig_y_np)
        mapped_x_np = topilimg(mapped_x_np)
        mapped_y_np = topilimg(mapped_y_np)

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, sharex='col', sharey='row'
        )
        f.suptitle('epoch {}'.format(e))

        ax1.imshow(orig_x_np)
        ax1.set_title('x')

        ax2.imshow(mapped_x_np)
        ax2.set_title('g(y) -> x')

        ax3.imshow(orig_y_np)
        ax3.set_title('y')

        ax4.imshow(mapped_y_np)
        ax4.set_title('f(x) -> y')

        f.savefig('visualize/{}.png'.format(e))

    def save(self, e, filename='zyklus.tar'):
        torch.save({
            'f_xy': self.f_xy.state_dict(),
            'g_yx': self.g_yx.state_dict(),
            'dx': self.dis_x.state_dict(),
            'dy': self.dis_y.state_dict(),
            'epoch': e + 1
        }, 'epoch{}_{}'.format(e, filename))
        print('Saved model state')

    def load(self, filedir):
        if os.path.isfile(filedir):
            checkpoint = torch.load(filedir)

            self.f_xy.load_state_dict(checkpoint['f_xy'])
            self.g_yx.load_state_dict(checkpoint['g_yx'])
            self.dis_x.load_state_dict(checkpoint['dx'])
            self.dis_y.load_state_dict(checkpoint['dy'])
            self.start_epoch = checkpoint['epoch']

            print('Model state loaded')

        else:
            print('Cant find file')

    def reset_gradients(self):
        """
        Helper function to reset gradients
        """
        self.f_xy.zero_grad()
        self.g_yx.zero_grad()
        self.dis_x.zero_grad()
        self.dis_y.zero_grad()

    def cudafy(self, t):
        """
        Helper function to cuda-fy our graphs
        """
        if self.cuda:
            if hasattr(t, 'cuda'):
                return t.cuda()
        return t
