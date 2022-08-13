from importlib import import_module

import torch.optim.lr_scheduler as lr_scheduler


class Optimizer:
    def __init__(self, args, target):
        # create optimizer
        # trainable = filter(lambda x: x.requires_grad, target.parameters())
        trainable = target.parameters()
        optimizer_name = args.optimizer
        lr = args.lr
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        # create scheduler
        if args.lr_scheduler == "multi_step":
            gamma = 0.5
            milestones = [200, 400]
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        elif args.lr_scheduler == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.end_epoch, eta_min=1e-5)
        else:
            raise NotImplementedError

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()
