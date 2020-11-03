import os
import torch
import csv
import random
import scipy.io
import numpy as np
from itertools import cycle
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as tf
from torchvision import transforms


class SIDD_Tensor(Dataset):

    def __init__(self, opt, csv_file_name, cpu_transform=None, gpu_transform=None, manual_length=None):
        self.opt = opt
        self.csv_file_name = csv_file_name
        self.cpu_transform = cpu_transform
        self.gpu_transform = gpu_transform
        self.manual_length = manual_length
        self.images_tensor = torch.Tensor()

        if os.path.exists(self.opt.dataset_out + self.get_file_name() + ".pt"):
            if opt.verbose:
                print("Loading Pre-Transformed Dataset...")
            self.images_tensor = torch.load(self.opt.dataset_out + self.get_file_name() + ".pt")
        else:
            print("Preparing Dataset...")
            csv_file_path = self.opt.dataset_root + csv_file_name
            with open(csv_file_path, newline='\n') as csv_file:
                line_reader = csv.reader(csv_file)
                lines = list(line_reader)
                line_pool = cycle(lines)
                with tqdm(total=self.__len__(), ncols=60) as pbar:
                    for idx, line in enumerate(line_pool):
                        if idx * 2 >= self.__len__():
                            break
                        scene_number = line[0].split("_")[0]
                        gt_0_path = self.opt.dataset_root + "Data/" + line[0] + "/" + scene_number + "_GT_SRGB_010.PNG"
                        noisy_0_path = self.opt.dataset_root + "Data/" + line[
                            0] + "/" + scene_number + "_NOISY_SRGB_010.PNG"
                        gt_1_path = self.opt.dataset_root + "Data/" + line[0] + "/" + scene_number + "_GT_SRGB_011.PNG"
                        noisy_1_path = self.opt.dataset_root + "Data/" + line[
                            0] + "/" + scene_number + "_NOISY_SRGB_011.PNG"

                        gt_0 = Image.open(gt_0_path)
                        noisy_0 = Image.open(noisy_0_path)
                        gt_1 = Image.open(gt_1_path)
                        noisy_1 = Image.open(noisy_1_path)

                        if self.cpu_transform:
                            data0 = self.cpu_transform({'gt': gt_0, 'noisy': noisy_0})
                            data1 = self.cpu_transform({'gt': gt_1, 'noisy': noisy_1})

                        pair_0 = torch.stack([data0['gt'], data0['noisy']], 0)
                        pair_1 = torch.stack([data1['gt'], data1['noisy']], 0)
                        self.images_tensor = torch.cat((self.images_tensor, torch.stack([pair_0, pair_1], 0)))
                        pbar.update(2)

            torch.save(self.images_tensor, self.opt.dataset_out + self.get_file_name() + ".pt")

    def __len__(self):
        return self.manual_length if self.manual_length is not None else (self.opt.batch_size * self.opt.epochs)

    def __getitem__(self, idx):
        if self.gpu_transform:
            gt0 = self.images_tensor[idx][0]
            noisy0 = self.images_tensor[idx][1]
            second_idx = random.randint(0, self.images_tensor.shape[0] - 1)
            gt1 = self.images_tensor[second_idx][0]
            noisy1 = self.images_tensor[second_idx][1]
            data = self.gpu_transform({'gt0': gt0, 'noisy0': noisy0, 'gt1': gt1, 'noisy1': noisy1})
            gt, noisy = data['gt0'], data['noisy0']
        else:
            gt, noisy = self.images_tensor[idx][0], self.images_tensor[idx][1]

        return {'gt': gt, 'noisy': noisy}

    def get_file_name(self):
        return str(self.__class__.__name__) + "_" + self.csv_file_name.split(".")[0] + "_PS" + str(
            self.opt.patch_size) + "_CNT" + str(self.__len__())

    def name(self):
        augs_str = ""
        for aug in self.opt.augs:
            augs_str += '_'
            augs_str += aug

        return str(self.__class__.__name__) + "_" + self.csv_file_name.split(".")[0] + "_PS" + str(
            self.opt.patch_size) + "_CNT" + str(self.__len__()) + "_AUGS" + augs_str


class SIDD_Validation(Dataset):
    def __init__(self, data_root, gt_file_name, noisy_file_name):
        self.data_root = data_root
        self.gt_file_name = gt_file_name
        self.noisy_file_name = noisy_file_name
        self.to_tensor = transforms.ToTensor()
        self.gt_blocks = scipy.io.loadmat(self.data_root + self.gt_file_name + '.mat')[self.gt_file_name]
        self.noisy_blocks = scipy.io.loadmat(self.data_root + self.noisy_file_name + '.mat')[self.noisy_file_name]
        shape = (len(self.gt_blocks) * len(self.gt_blocks[0]), 256, 256, 3)
        self.gt_blocks = np.reshape(self.gt_blocks, shape)
        self.noisy_blocks = np.reshape(self.noisy_blocks, shape)

    def __len__(self):
        return len(self.gt_blocks)

    def __getitem__(self, idx):
        gt = self.to_tensor(self.gt_blocks[idx])
        noisy = self.to_tensor(self.noisy_blocks[idx])
        return {'gt': gt, 'noisy': noisy}
