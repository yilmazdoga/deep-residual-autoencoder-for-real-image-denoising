import os
import time
import shutil
import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchsummary import summary
from prefetch_generator import BackgroundGenerator
from options.train_options import TrainOptions
from data.data_loader import create_data_loader
from models import models, losses
from models import util_funcs
from scripts.utils.utils import clean_previous_logs

if __name__ == '__main__':
    opt = TrainOptions().parse()

    training_name = '_'.join(
        [opt.model, opt.loss_func, opt.dataset, str(opt.augs), str(opt.batch_size), str(opt.patch_size),
         str(opt.epochs)])

    train_loader = create_data_loader('train', opt=opt)
    test_loader = create_data_loader('test', opt=opt, shuffle=False)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.gpus > 0) else "cpu")

    if opt.verbose:
        print("device:", device)

    net = getattr(models, opt.model)()
    summary(net, (3, opt.patch_size, opt.patch_size), device="cpu")

    if (device.type == 'cuda') and (opt.gpus > 1):
        if opt.verbose:
            print("DataParallel Using", list(range(opt.gpus)), "GPUs")
        net = torch.nn.DataParallel(net, list(range(opt.gpus)))

    net.to(device)
    net.apply(util_funcs.weights_init)

    criterion = getattr(losses, opt.loss_func)().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(opt.epochs * 0.6), int(opt.epochs * 0.9)], gamma=0.1)

    if not opt.no_tb:
        training_log_dir = "runs/" + training_name + "_training"
        validation_log_dir = "runs/" + training_name + "_validation"

        clean_previous_logs(opt, training_log_dir)
        clean_previous_logs(opt, validation_log_dir)

        tb_training = SummaryWriter(training_log_dir)
        tb_validation = SummaryWriter(validation_log_dir)

    net.train()
    pbar = tqdm(total=opt.epochs, ncols=60)
    for epoch in range(opt.epochs):
        running_loss = 0
        for i, data in enumerate(BackgroundGenerator(train_loader)):
            start_time = time.time()

            gt, noisy = data['gt'].to(device), data['noisy'].to(device)

            prepare_time = start_time - time.time()

            optimizer.zero_grad()
            denoised = net(noisy)
            loss = criterion(denoised, gt)
            loss.backward()
            optimizer.step()

            process_time = start_time - time.time() - prepare_time
            running_loss += loss.item()

        scheduler.step()
        pbar.update(1)
        if not opt.no_tb:
            tb_training.add_scalar('loss', running_loss / len(train_loader), global_step=epoch)
            tb_training.add_scalar('prepare_time', prepare_time, global_step=epoch)
            tb_training.add_scalar('process_time', process_time, global_step=epoch)

        if epoch % 10 == 0 or epoch == opt.epochs - 1:
            net.eval()
            with torch.no_grad():
                validation_loss = 0
                for i, data in enumerate(BackgroundGenerator(test_loader)):
                    gt, noisy = data['gt'].to(device), data['noisy'].to(device)
                    denoised = net(noisy)
                    loss = criterion(denoised, gt)
                    validation_loss += loss.item()

                tb_validation.add_scalar('loss', validation_loss / len(test_loader), global_step=epoch)

                if opt.sample_img_path is not None:
                    noisy = Image.open(opt.sample_img_path)
                    noisy = transforms.ToTensor()(noisy).to(device)
                    denoised = net(noisy[None, ...])
                    tb_validation.add_image('sample_image', denoised[0], global_step=epoch, dataformats='CHW')

            net.train()

    save_path = opt.checkpoints_dir + training_name + '.pth'
    if (device.type == 'cuda') and (opt.gpus > 1):
        torch.save(net.module.state_dict(), save_path)
    else:
        torch.save(net.state_dict(), save_path)
    pbar.set_description("Training done. State-Dict saved. ")
    pbar.close()
