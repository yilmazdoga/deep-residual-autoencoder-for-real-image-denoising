from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.is_train = True
        self.world_size = 0

    def initialize(self):
        BaseOptions.initialize(self)

        # for training
        self.parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight Decay Rate.')
        self.parser.add_argument('--no-tb', action='store_true', default=False,
                                 help='if specified, use tensorboard logging. Requires tensorboard installed')
        self.parser.add_argument('--sample-img-path', type=str,
                                 default='datasets/SIDD_Dataset/Sample_Images/Noisy_Images/NOISY_0.png',
                                 help='sample image path')
        self.parser.add_argument('--gpus', type=int, default=1, help='use 0 for CPU')
