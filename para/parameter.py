import argparse


class Parameter:
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser(
            description='Intermediate Frames Extracting From Rolling Shutter Distortion')

        # global parameters
        self.parser.add_argument('--seed', type=int, default=39, help='random seed')
        self.parser.add_argument('--threads', type=int, default=4, help='# of threads for dataloader')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
        self.parser.add_argument('--no_profile', action='store_true', help='show # of parameters and computation cost')
        self.parser.add_argument('--profile_H', type=int, default=720,
                                 help='height of image to generate profile of model')
        self.parser.add_argument('--profile_W', type=int, default=1280,
                                 help='width of image to generate profile of model')
        self.parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_file', type=str, default='', help='the path of checkpoint file for resume')

        # data parameters
        self.parser.add_argument('--data_root', type=str, default='/home/zhong/Dataset/', help='the path of dataset')
        self.parser.add_argument('--dataset', type=str, default='RS-GOPRO_DS', help='dataset name')
        self.parser.add_argument('--save_dir', type=str, default='./experiment/',
                                 help='directory to save logs of experiments')
        self.parser.add_argument('--frames', type=int, default=5,
                                 help='number of gs frames extracted from rs img [1|3|5||9]')
        self.parser.add_argument('--patch_size', type=int, nargs='*', default=[256, 256])

        # model parameters
        self.parser.add_argument('--model', type=str, default='DIFE_MulCatFusion', help='type of model to construct')
        self.parser.add_argument('--n_feats', type=int, default=32, help='base # of channels for Conv')
        self.parser.add_argument('--future_frames', type=int, default=1, help='use # of future frames')
        self.parser.add_argument('--past_frames', type=int, default=1, help='use # of past frames')

        # loss parameters
        self.parser.add_argument('--loss', type=str, default='1*Charbonnier|1e-1*Perceptual|1e-1*Variation',
                                 help='type of loss function, e.g. 1*Charbonnier|1e-1*Perceptual|1e-1*Variation|1e-2*EPE')

        # metrics parameters
        self.parser.add_argument('--metrics', type=str, default='PSNR', help='type of evaluation metrics')

        # optimizer parameters
        self.parser.add_argument('--optimizer', type=str, default='AdamW', help='method of optimization')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--lr_scheduler', type=str, default='cosine',
                                 help='learning rate adjustment stratedy')
        self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')

        # training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=500, help='last epoch number')
        self.parser.add_argument('--trainer_mode', type=str, default='dp',
                                 help='trainer mode: distributed data parallel (ddp) or data parallel (dp)')

        # test parameters
        self.parser.add_argument('--test_only', action='store_true', help='only do test')
        self.parser.add_argument('--test_frames', type=int, default=3,
                                 help='frame size for test, if GPU memory is small, please reduce this value')
        self.parser.add_argument('--test_save_dir', type=str, help='where to save test results')
        self.parser.add_argument('--test_checkpoint', type=str,
                                 default='./model/checkpoints/model_best.pth.tar',
                                 help='the path of checkpoint file for test')
        self.parser.add_argument('--video', action='store_true', help='if true, generate video results')

        args, _ = self.parser.parse_known_args()

        args.normalize = True
        args.centralize = False

        return args
