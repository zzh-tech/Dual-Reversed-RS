from importlib import import_module

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        model_name = args.model
        self.module = import_module('model.{}'.format(model_name))
        self.model = self.module.Model(args)

    def forward(self, iter_samples, calc_loss=False, return_velocity=False):
        return self.module.inference(self.model, iter_samples, calc_loss, return_velocity)

    def profile(self):
        H, W = self.args.profile_H, self.args.profile_W
        seq_length = self.args.future_frames + self.args.past_frames + 1
        flops, params = self.module.cost_profile(self.model, H, W, seq_length, self.args.frames)
        return flops / self.args.frames, params
