from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def __init__(self):
        super().__init__()
        self.is_train = False

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results-dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--sidd-validation-root', type=str, default="datasets/SIDD_Dataset/SIDD_Validation/",
                                 help='Validation dataset root')
        self.parser.add_argument('--gt-image-path', type=str,
                                 default='datasets/SIDD_Dataset/Sample_Images/GT_Images/GT_0.png',
                                 help='input gt image')
        self.parser.add_argument('--noisy-image-path', type=str,
                                 default='datasets/SIDD_Dataset/Sample_Images/Noisy_Images/NOISY_0.png',
                                 help='input noisy image')
