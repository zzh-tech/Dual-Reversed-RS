import os
import random
import time
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from data import Data
from model import Model
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util import Logger
from model.loss import loss_parse
from .optimizer import Optimizer
from .utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process(args):
    """
    data parallel training
    """
    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create logger
    logger = Logger(args)
    logger.writer = SummaryWriter(logger.save_dir)

    # create model
    logger('building {} model ...'.format(args.model), prefix='\n')
    model = Model(args).to(device)
    logger('model structure:', model, verbose=False)

    # create measurement according to metrics
    metrics_name = args.metrics
    module = import_module('train.metrics')
    val_range = 2.0 ** 8 - 1
    metrics = getattr(module, metrics_name)(centralize=args.centralize, normalize=args.normalize,
                                            val_range=val_range).to(device)

    # create optimizer
    opt = Optimizer(args, model)

    # distributed data parallel
    model = nn.DataParallel(model)

    # record model profile: computation cost & # of parameters
    if not args.no_profile:
        profile_model = Model(args).to(device)
        flops, params = profile_model.profile()
        logger('generating profile of {} model ...'.format(args.model), prefix='\n')
        logger('[profile] computation cost: {:.2f} GMACs, parameters: {:.2f} M'.format(
            flops / 10 ** 9, params / 10 ** 6), timestamp=False)
        del profile_model

    # create dataloader
    logger('loading {} dataloader ...'.format(args.dataset), prefix='\n')
    data = Data(args, device_id=0)
    train_loader = data.dataloader_train
    valid_loader = data.dataloader_valid

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume_file):
            checkpoint = torch.load(args.resume_file, map_location=lambda storage, loc: storage.cuda(0))
            logger('loading checkpoint {} ...'.format(args.resume_file))
            logger.register_dict = checkpoint['register_dict']
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            opt.optimizer.load_state_dict(checkpoint['optimizer'])
            opt.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            msg = 'no check point found at {}'.format(args.resume_file)
            logger(msg, verbose=False)
            raise FileNotFoundError(msg)

    # training and validation
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train(train_loader, model, metrics, opt, epoch, args, logger)
        valid(valid_loader, model, metrics, epoch, args, logger)

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'register_dict': logger.register_dict,
            'optimizer': opt.optimizer.state_dict(),
            'scheduler': opt.scheduler.state_dict()
        }
        logger.save(checkpoint)


def train(train_loader, model, metrics, opt, epoch, args, logger):
    # training mode
    model.train()
    logger('[Epoch {} / lr {:.2e}]'.format(epoch, opt.get_lr()), prefix='\n')

    losses_meter = {}
    _, losses_name = loss_parse(args.loss)
    losses_name.append('all')
    for key in losses_name:
        losses_meter[key] = AverageMeter()

    measure_meter = AverageMeter()
    batchtime_meter = AverageMeter()

    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(train_loader), ncols=80)

    for iter_samples in train_loader:
        # forward
        for (key, val) in enumerate(iter_samples):
            iter_samples[key] = val.to(device, non_blocking=True)
        inputs = iter_samples[0]
        labels = iter_samples[1]
        outputs, losses = model(iter_samples, calc_loss=True)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        measure = metrics(outputs.detach(), labels)

        # record value of losses and measurements
        for key in losses_name:
            losses_meter[key].update(losses[key].detach().item(), inputs.size(0))
        measure_meter.update(measure.detach().item(), inputs.size(0))

        # backward and optimize
        opt.zero_grad()
        losses['all'].backward()

        # clip the grad
        clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

        # update weights
        opt.step()

        # measure elapsed time
        batchtime_meter.update(time.time() - end)
        end = time.time()
        pbar.update(args.batch_size)

    pbar.close()

    # record info
    logger.register(args.loss + '_train', epoch, losses_meter['all'].avg)
    logger.register(args.metrics + '_train', epoch, measure_meter.avg)
    for key in losses_name:
        logger.writer.add_scalar(key + '_loss_train', losses_meter[key].avg, epoch)
    logger.writer.add_scalar(args.metrics + '_train', measure_meter.avg, epoch)
    logger.writer.add_scalar('lr', opt.get_lr(), epoch)

    # show info
    logger('[train] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end - start, batchtime_meter.avg),
           timestamp=False)
    logger.report([[args.loss, 'min'], [args.metrics, 'max']], state='train', epoch=epoch)
    msg = '[train]'
    for key, meter in losses_meter.items():
        if key == 'all':
            continue
        msg += ' {} : {:4f};'.format(key, meter.avg)
    logger(msg, timestamp=False)

    # adjust learning rate
    opt.lr_schedule()


def valid(valid_loader, model, metrics, epoch, args, logger):
    # evaluation mode
    model.eval()

    with torch.no_grad():
        losses_meter = {}
        _, losses_name = loss_parse(args.loss)
        losses_name.append('all')
        for key in losses_name:
            losses_meter[key] = AverageMeter()

        measure_meter = AverageMeter()
        batchtime_meter = AverageMeter()
        start = time.time()
        end = time.time()
        pbar = tqdm(total=len(valid_loader), ncols=80)

        for iter_samples in valid_loader:
            for (key, val) in enumerate(iter_samples):
                iter_samples[key] = val.to(device, non_blocking=True)
            inputs = iter_samples[0]
            labels = iter_samples[1]
            outputs, losses = model(iter_samples, calc_loss=True)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            measure = metrics(outputs.detach(), labels)
            for key in losses_name:
                losses_meter[key].update(losses[key].detach().item(), inputs.size(0))
            measure_meter.update(measure.detach().item(), inputs.size(0))

            batchtime_meter.update(time.time() - end)
            end = time.time()
            pbar.update(args.batch_size)

    pbar.close()

    # record info
    logger.register(args.loss + '_valid', epoch, losses_meter['all'].avg)
    logger.register(args.metrics + '_valid', epoch, measure_meter.avg)
    for key in losses_name:
        logger.writer.add_scalar(key + '_loss_valid', losses_meter[key].avg, epoch)
    logger.writer.add_scalar(args.metrics + '_valid', measure_meter.avg, epoch)

    # show info
    logger('[valid] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end - start, batchtime_meter.avg),
           timestamp=False)
    logger.report([[args.loss, 'min'], [args.metrics, 'max']], state='valid', epoch=epoch)
    msg = '[valid]'
    for key, meter in losses_meter.items():
        if key == 'all':
            continue
        msg += ' {} : {:4f};'.format(key, meter.avg)
    logger(msg, timestamp=False)
