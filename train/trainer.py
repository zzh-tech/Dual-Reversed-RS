from datetime import datetime
from util import Logger
from .dp import process
from .test import test


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def run(self):
        # recoding parameters
        self.args.time = datetime.now()
        logger = Logger(self.args)
        logger.record_para()

        # training
        if not self.args.test_only:
            process(self.args)

        # test
        test(self.args, logger)
