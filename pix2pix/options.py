import argparse
import os
import utils


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument(
            '--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument(
            '--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument(
            '--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument(
            '--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument(
            '--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument(
            '--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument(
            '--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str,
                                 default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str,
                                 default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument(
            '--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument(
            '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='rarepepes',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--align_data', action='store_true',
                                 help='if True, the datasets are loaded from "test" and "train" directories and the data pairs are aligned')
        self.parser.add_argument(
            '--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument(
            '--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument(
            '--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument(
            '--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--identity', type=float, default=0.0,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument(
            '--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float(
            "inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument(
            '--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument(
            '--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')

        # NOT-IMPLEMENTED self.parser.add_argument('--preprocessing', type=str,
        # default='resize_and_crop', help='resizing/cropping strategy')
        self.isTrain = True


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument(
            '--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument(
            '--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument(
            '--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--how_many', type=int, default=3, help='how many test images to run')
        self.isTrain = False
