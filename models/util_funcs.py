from torch import nn
import torch


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)


def validation_loss(dvc, model_, criterion_, validation_data):
    model_.eval()
    with torch.no_grad():
        loss_ = 0
        for i_, data_ in enumerate(validation_data):
            GT_, NOISY_ = data_
            GT_, NOISY_ = GT_.to(dvc, non_blocking=True), NOISY_.to(dvc, non_blocking=True)
            DENOISED_ = model_(NOISY_)
            loss_ = criterion_(DENOISED_, GT_)
            loss_ += loss_.item()
    model_.train()
    return loss_ / len(validation_data)
