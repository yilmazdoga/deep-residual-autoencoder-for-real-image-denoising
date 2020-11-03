import argparse
import os


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = self.parser

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='DN_Transformer',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints/', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='AEv2_0', help='which model to use')
        self.parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. (default: 0.002)')
        self.parser.add_argument('--loss-func', type=str, default="L1_MSSSIM", help='Loss Function.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train. (default: 200)')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes
        self.parser.add_argument('--batch-size', type=int, default=24, help='input batch size')
        self.parser.add_argument('--patch-size', type=int, default=128, help='crop to this size, -1 to disable crop.')

        # for setting inputs
        self.parser.add_argument('--dataset', type=str, default="SIDD_Tensor", help='Dataset name')
        self.parser.add_argument('--dataset-root', type=str, default="datasets/SIDD_Dataset/SIDD_Medium_sRGB/",
                                 help='Dataset root')
        self.parser.add_argument('--dataset-out', type=str, default="data/prepared_data/",
                                 help='prepared dataset folder')
        self.parser.add_argument('--test-set-length', type=int, default=64, help='test set length')
        self.parser.add_argument('--augs', type=str, default='MixUp_Tensor', help='use -1 for no aug.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train

        augs = self.opt.augs.split(',')
        self.opt.augs = []
        for aug in augs:
            self.opt.augs.append(aug)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
