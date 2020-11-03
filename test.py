import os
import sys
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from piq import ssim
from piq import psnr
from models import models
from data.data_loader import create_data_loader
from options.test_options import TestOptions
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()

    training_name = '_'.join(
        [opt.model, opt.loss_func, opt.dataset, str(opt.augs), str(opt.batch_size), str(opt.patch_size),
         str(opt.epochs)])

    if opt.verbose:
        print("Training Name:", training_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.verbose:
        print("Device:", device)

    net = getattr(models, opt.model)()
    summary(net, (3, opt.patch_size, opt.patch_size), device="cpu")
    net = net.to(device)

    path_to_weights = "./checkpoints/" + training_name + ".pth"
    if os.path.isfile(path_to_weights):
        net.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))
        net.to(device)
    else:
        print("no pre-trained model found with current parameters")
        sys.exit(0)

    net.eval()
    with torch.no_grad():
        if opt.gt_image_path is None or opt.noisy_image_path is None:
            validation_loader = create_data_loader('validation', opt=opt, shuffle=False)
            sum_SSIM = 0
            sum_PSNR = 0
            pbar = tqdm(total=len(validation_loader), ncols=60)
            for i, data in enumerate(validation_loader):
                gt, noisy = data['gt'].to(device), data['noisy'].to(device)
                denoised = net(noisy)
                sum_SSIM += ssim(denoised.to('cpu'), gt.to('cpu'), data_range=1.).item()
                sum_PSNR += psnr(denoised.to('cpu'), gt.to('cpu'), data_range=1.).item()
                pbar.update(1)

            pbar.set_description("Test Finished.")
            pbar.close()
            average_SSIM = sum_SSIM / len(validation_loader)
            average_PSNR = sum_PSNR / len(validation_loader)
            print("Average SSIM:", str(average_SSIM))
            print("Average PSNR:", str(average_PSNR))
            with open(opt.results_dir + training_name + '_results.txt', 'w') as f:
                print("SSIM", file=f)
                print(average_SSIM, file=f)
                print("PSNR", file=f)
                print(average_PSNR, file=f)

        else:
            gt = Image.open(opt.gt_image_path)
            noisy = Image.open(opt.noisy_image_path)
            _noisy = transforms.ToTensor()(noisy)
            _noisy = _noisy.to(device)
            _denoised = net(_noisy[None, ...])
            denoised = transforms.ToPILImage()(_denoised[0])

            # denoised.save("denoised.png")
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.axis("off")
            ax1.set_title("GT", fontsize=12)
            ax1.margins(0, 0)
            ax1.imshow(gt)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.axis("off")
            ax2.set_title("NOISY", fontsize=12)
            ax2.margins(0, 0)
            ax2.imshow(noisy)
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.axis("off")
            ax3.set_title("DENOISED", fontsize=12)
            ax3.margins(0, 0)
            ax3.imshow(denoised)
            plt.show()
