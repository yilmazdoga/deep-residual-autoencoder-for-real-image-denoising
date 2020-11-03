from torch import nn
from pytorch_msssim import MS_SSIM as _MS_SSIM
from pytorch_msssim import SSIM as _SSIM


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        l1_loss = self.l1(x, y)

        return l1_loss


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
        self.l2 = nn.MSELoss()

    def forward(self, x, y):
        l2_loss = self.l2(x, y)

        return l2_loss


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.ssim = _SSIM(data_range=1, size_average=True, win_size=7, channel=3)

    def forward(self, x, y):
        ssim_loss = 1 - self.ssim(x, y)

        return ssim_loss


class MSSSIM(nn.Module):
    def __init__(self):
        super(MSSSIM, self).__init__()
        self.ms_ssim = _MS_SSIM(data_range=1, size_average=True, win_size=7, channel=3)

    def forward(self, x, y):
        ms_ssim_loss = 1 - self.ms_ssim(x, y)

        return ms_ssim_loss


class L1_SSIM(nn.Module):
    def __init__(self):
        super(L1_SSIM, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = _SSIM(data_range=1, size_average=True, win_size=7, channel=3)

    def forward(self, x, y):
        ssim_loss = 1 - self.ssim(x, y)
        l1_loss = self.l1(x, y)

        return l1_loss + ssim_loss


class L1_MSSSIM(nn.Module):
    def __init__(self):
        super(L1_MSSSIM, self).__init__()
        self.l1 = nn.L1Loss()
        self.ms_ssim = _MS_SSIM(data_range=1, size_average=True, win_size=7, channel=3)

    def forward(self, x, y):
        ms_ssim_loss = 1 - self.ms_ssim(x, y)
        l1_loss = self.l1(x, y)

        return l1_loss + ms_ssim_loss
