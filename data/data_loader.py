import torch
from data import data_sets
from data import transforms as _transforms
from torchvision import transforms
from torch.utils.data import DataLoader


def create_data_loader(mode, opt, shuffle=True):
    if opt.dataset == 'SIDD_Tensor':

        if mode == 'train':
            csv_file_name = "train.txt"

            cpu_transforms = []
            if opt.patch_size != '-1':
                cpu_transforms.append(_transforms.RandomCrop(opt.patch_size))
            cpu_transforms.append(_transforms.ToTensor())

            gpu_transforms = []
            if opt.augs[0] != '-1':
                for aug in opt.augs:
                    gpu_transforms.append(getattr(_transforms, aug)())

            dataset = data_sets.SIDD_Tensor(opt=opt, csv_file_name=csv_file_name,
                                            cpu_transform=transforms.Compose(cpu_transforms),
                                            gpu_transform=transforms.Compose(gpu_transforms))

            return DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

        elif mode == 'test':
            csv_file_name = "test.txt"

            cpu_transforms = []
            if opt.patch_size != '-1':
                cpu_transforms.append(_transforms.RandomCrop(opt.patch_size))
            cpu_transforms.append(_transforms.ToTensor())

            dataset = data_sets.SIDD_Tensor(opt=opt, csv_file_name=csv_file_name,
                                            cpu_transform=transforms.Compose(cpu_transforms),
                                            manual_length=opt.test_set_length)

            return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=4, pin_memory=True)

        elif mode == 'validation':
            dataset = data_sets.SIDD_Validation(data_root=opt.sidd_validation_root,
                                                gt_file_name='ValidationGtBlocksSrgb',
                                                noisy_file_name='ValidationNoisyBlocksSrgb')

            return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=4, pin_memory=True)
        else:
            raise Exception("dataloader mode incorrect")

    else:
        print("Dataset not found.")
        return None
