import data_sets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data.transforms import MixUp_Tensor, CutMix_Tensor, CutOut_Tensor, RandomCrop, ToTensor

from options.test_options import TestOptions
from data.data_loader import create_data_loader

opt = TestOptions().parse()
opt.dataset_root = "../datasets/SIDD_Dataset/SIDD_Medium_sRGB/"
opt.sidd_validation_root = '../' + opt.sidd_validation_root
opt.dataset_out = "./prepared_data/"

data_loader = create_data_loader('validation', opt=opt)

print('length:', len(data_loader))

data = next(iter(data_loader))
GT, NOISY = data['gt'], data['noisy']
GT_as_PIL = transforms.ToPILImage()(GT[0])
NOISY_as_PIL = transforms.ToPILImage()(NOISY[0])
GT_as_np = np.rot90(np.transpose(GT.numpy()[0]), 3)
NOISY_as_np = np.rot90(np.transpose(NOISY.numpy()[0]), 3)
# GT_as_PIL.save("gt_test.png")
# NOISY_as_PIL.save("noisy_test.png")
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis("off")
ax1.set_title("GT", fontsize=12)
ax1.margins(0, 0)
ax1.imshow(GT_as_np)
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis("off")
ax2.set_title("NOISY", fontsize=12)
ax2.margins(0, 0)
ax2.imshow(NOISY_as_np)
plt.show()
