import torch
import random
from torchvision import transforms
from torchvision.transforms import functional as tf


class RandomCrop(object):
    """Crop randomly image pair."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        gt, noisy = data['gt'], data['noisy']

        i, j, h, w = transforms.RandomCrop.get_params(gt, output_size=self.output_size)

        gt = tf.crop(gt, i, j, h, w)
        noisy = tf.crop(noisy, i, j, h, w)

        return {'gt': gt, 'noisy': noisy}


class ToTensor(object):
    """Convert PIL to Tensor."""

    def __call__(self, data):
        gt, noisy = data['gt'], data['noisy']

        func = transforms.ToTensor()

        return {'gt': func(gt), 'noisy': func(noisy)}


class MixUp_Tensor(object):
    """Apply MixUp Augmentation."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, data):
        gt0, noisy0, gt1, noisy1 = data['gt0'], data['noisy0'], data['gt1'], data['noisy1']

        if random.uniform(0, 1) < self.prob:
            mix_ratio = random.uniform(0.0, 1.0)
            gt = (gt0 * mix_ratio) + (gt1 * (1 - mix_ratio))
            noisy = (noisy0 * mix_ratio) + (noisy1 * (1 - mix_ratio))
        else:
            gt, noisy = gt0, noisy0

        return {'gt0': gt, 'noisy0': noisy, 'gt1': gt1, 'noisy1': noisy1}


class CutMix_Tensor(object):
    """Apply CutMix Augmentation."""

    def __init__(self, prob=0.2, max_crop=0.5):
        self.prob = prob
        self.max_crop = max_crop

    def __call__(self, data):

        gt0, noisy0, gt1, noisy1 = data['gt0'], data['noisy0'], data['gt1'], data['noisy1']

        if random.uniform(0, 1) < self.prob:

            crop_ratio = random.uniform(0.2, self.max_crop)
            i, j, h, w = transforms.RandomCrop.get_params(gt0, output_size=(
                int(gt1.shape[1] * crop_ratio), int(gt1.shape[2] * crop_ratio)))

            mask = torch.zeros_like(gt1).bool()
            mask[:, i:i + h, j:j + w] = True
            inverse_mask = ~mask

            gt = (gt0 * inverse_mask.int()) + (gt1 * mask.int())
            noisy = (noisy0 * inverse_mask.int()) + (noisy1 * mask.int())
        else:
            gt, noisy = gt0, noisy0

        return {'gt0': gt, 'noisy0': noisy, 'gt1': gt1, 'noisy1': noisy1}


class CutOut_Tensor(object):
    """Apply CutOut Augmentation."""

    def __init__(self, prob=0.2, max_crop=0.5):
        self.prob = prob
        self.max_crop = max_crop

    def __call__(self, data):
        gt0, noisy0, gt1, noisy1 = data['gt0'], data['noisy0'], data['gt1'], data['noisy1']

        if random.uniform(0, 1) < self.prob:

            crop_ratio = random.uniform(0.2, self.max_crop)
            i, j, h, w = transforms.RandomCrop.get_params(gt0, output_size=(
                int(gt0.shape[1] * crop_ratio), int(gt0.shape[2] * crop_ratio)))

            mask = torch.zeros_like(gt0).bool()
            mask[:, i:i + h, j:j + w] = True
            inverse_mask = ~mask

            gt0 = gt0 * inverse_mask.int()
            noisy0 = noisy0 * inverse_mask.int()
        else:
            return {'gt0': gt0, 'noisy0': noisy0, 'gt1': gt1, 'noisy1': noisy1}

        return {'gt0': gt0, 'noisy0': noisy0, 'gt1': gt1, 'noisy1': noisy1}
