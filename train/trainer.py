import os
from datetime import datetime

import torch.multiprocessing as mp

from util import Logger
from .ddp import dist_process
from .dp import process
from .test import test


class Trainer(object):
    def __init__(self, args):
        self.args = args
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '6666'

    def run(self):
        # recoding parameters
        self.args.time = datetime.now()
        logger = Logger(self.args)
        logger.record_para()

        # training
        if not self.args.test_only:
            if self.args.trainer_mode == 'ddp':
                gpus = self.args.num_gpus
                processes = []
                for rank in range(gpus):
                    p = mp.Process(target=dist_process, args=(rank, self.args))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            elif self.args.trainer_mode == 'dp':
                process(self.args)

        # test
        test(self.args, logger)
