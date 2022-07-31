import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    from .warplayer import multi_warp
except:
    from model.warplayer import multi_warp

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )  # TODO WARNING: Notable change (No Batch Norm)


class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64, num_flows=3, mode='backward'):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2 * c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
        )
        if mode == 'backward':
            self.conv1 = nn.ConvTranspose2d(2 * c, 2 * num_flows * 2, 4, 2, 1)  # TODO WARNING: Notable change
        elif mode == 'forward':
            self.conv1 = nn.ConvTranspose2d(2 * c, 2 * num_flows * 3, 4, 2, 1)  # TODO WARNING: Notable change
        else:
            raise ValueError

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                              align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x)
        flow = self.conv1(x)
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False)
        return flow


class IFBlockSgl(nn.Module):
    def __init__(self, in_planes, scale=1, c=64, num_flows=3):
        super(IFBlockSgl, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2 * c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
        )
        self.conv1 = nn.ConvTranspose2d(2 * c, num_flows * 2, 4, 2, 1)  # TODO WARNING: Notable change

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                              align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x)
        flow = self.conv1(x)
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False)
        return flow


class FlowNet(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNet, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6 + 2 * num_flows, scale=8, c=192, num_flows=num_flows)
        self.block1 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.block2 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.block3 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=1, c=48, num_flows=num_flows)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)

        flow0 = self.block0(torch.cat((x, encoding), dim=1))
        F1 = flow0
        encoding = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                 recompute_scale_factor=False)
        F1 = self._mul_encoding(F1, encoding)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F1_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F1_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2 = self._mul_encoding(F2, encoding)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F2_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F2_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3 = self._mul_encoding(F3, encoding)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F3_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F3_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4 = self._mul_encoding(F4, encoding)

        return F4, [F1, F2, F3, F4]


class FlowNetSgl(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNetSgl, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlockSgl(in_planes=6 + num_flows, scale=8, c=192, num_flows=num_flows)
        self.block1 = IFBlockSgl(in_planes=(2 + 3 + 1) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.block2 = IFBlockSgl(in_planes=(2 + 3 + 1) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.block3 = IFBlockSgl(in_planes=(2 + 3 + 1) * num_flows, scale=1, c=48, num_flows=num_flows)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)
        encoding = encoding[:, :self.num_flows]

        flow0 = self.block0(torch.cat((x, encoding), dim=1))
        F1 = flow0
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)
        F1 = self._mul_encoding(F1, encoding_ds)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_imgs = multi_warp(x_t2b, F1_large)  # (n, num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, encoding, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2 = self._mul_encoding(F2, encoding_ds)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_imgs = multi_warp(x_t2b, F2_large)
        flow2 = self.block2(torch.cat((warped_imgs, encoding, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3 = self._mul_encoding(F3, encoding_ds)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_imgs = multi_warp(x_t2b, F3_large)
        flow3 = self.block3(torch.cat((warped_imgs, encoding, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4 = self._mul_encoding(F4, encoding_ds)

        return F4, [F1, F2, F3, F4]


class FlowNetWoprior(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNetWoprior, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6, scale=8, c=192, num_flows=num_flows)
        self.block1 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.block2 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.block3 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=1, c=48, num_flows=num_flows)

    def forward(self, x, encoding):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)

        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F1_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F1_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F2_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F2_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F3_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F3_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)

        return F4, [F1, F2, F3, F4]


class FlowNetMulCat(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNetMulCat, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6 + 2 * num_flows, scale=8, c=192, num_flows=num_flows)
        self.block1 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.block2 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.block3 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=1, c=48, num_flows=num_flows)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)

        flow0 = self.block0(torch.cat((x, encoding), dim=1))
        F1 = flow0
        F1 = self._mul_encoding(F1, encoding_ds)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F1_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F1_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, encoding, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2 = self._mul_encoding(F2, encoding_ds)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F2_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F2_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, encoding, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3 = self._mul_encoding(F3, encoding_ds)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F3_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F3_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, encoding, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4 = self._mul_encoding(F4, encoding_ds)

        return F4, [F1, F2, F3, F4]


class FlowNetMulCatFusion(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNetMulCatFusion, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6 + 2 * num_flows, scale=8, c=192, num_flows=num_flows)
        self.conv0 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block1 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.conv1 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block2 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.conv2 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block3 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=1, c=48, num_flows=num_flows)
        self.conv3 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding, return_velocity=False):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)

        flow0 = self.block0(torch.cat((x, encoding), dim=1))
        F1 = flow0
        F1 = self._mul_encoding(F1, encoding_ds)
        F1 = self.conv0(F1)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F1_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F1_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, encoding, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2 = self._mul_encoding(F2, encoding_ds)
        F2 = self.conv1(F2)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F2_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F2_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, encoding, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3 = self._mul_encoding(F3, encoding_ds)
        F3 = self.conv2(F3)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F3_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F3_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, encoding, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4 = self._mul_encoding(F4, encoding_ds)
        F4 = self.conv3(F4)

        if return_velocity:
            return F4, [F1, F2, F3, F4], flow0 + flow1 + flow2 + flow3
        return F4, [F1, F2, F3, F4]


class FlowNetMulMulFusion(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNetMulMulFusion, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6, scale=8, c=192, num_flows=num_flows)
        self.conv0 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block1 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.conv1 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block2 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.conv2 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block3 = IFBlock(in_planes=2 * (2 + 3) * num_flows, scale=1, c=48, num_flows=num_flows)
        self.conv3 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding, return_velocity=False):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)

        flow0 = self.block0(x)
        F1 = flow0
        F1 = self._mul_encoding(F1, encoding_ds)
        F1 = self.conv0(F1)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F1_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F1_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2 = self._mul_encoding(F2, encoding_ds)
        F2 = self.conv1(F2)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F2_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F2_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3 = self._mul_encoding(F3, encoding_ds)
        F3 = self.conv2(F3)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F3_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F3_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4 = self._mul_encoding(F4, encoding_ds)
        F4 = self.conv3(F4)

        if return_velocity:
            return F4, [F1, F2, F3, F4], flow0 + flow1 + flow2 + flow3
        return F4, [F1, F2, F3, F4]


class FlowNetMulCatFusionSS(nn.Module):
    def __init__(self, num_flows=3):
        super(FlowNetMulCatFusionSS, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6 + 2 * num_flows, scale=8, c=192, num_flows=num_flows, mode='forward')
        self.conv0 = conv_wo_act(2 * 3 * num_flows, 2 * 3 * num_flows, 1, 1, 0)
        self.block1 = IFBlock(in_planes=2 * (3 + 3 + 1) * num_flows, scale=4, c=128, num_flows=num_flows,
                              mode='forward')
        self.conv1 = conv_wo_act(2 * 3 * num_flows, 2 * 3 * num_flows, 1, 1, 0)
        self.block2 = IFBlock(in_planes=2 * (3 + 3 + 1) * num_flows, scale=2, c=96, num_flows=num_flows, mode='forward')
        self.conv2 = conv_wo_act(2 * 3 * num_flows, 2 * 3 * num_flows, 1, 1, 0)
        self.block3 = IFBlock(in_planes=2 * (3 + 3 + 1) * num_flows, scale=1, c=48, num_flows=num_flows, mode='forward')
        self.conv3 = conv_wo_act(2 * 3 * num_flows, 2 * 3 * num_flows, 1, 1, 0)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 3, h, w)
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)

        flow0 = self.block0(torch.cat((x, encoding), dim=1))
        F1 = flow0
        F1[:, :4 * self.num_flows] = self._mul_encoding(F1[:, :4 * self.num_flows], encoding_ds)
        F1 = self.conv0(F1)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0
        F1_large_t2b = torch.cat([F1_large[:, :2 * self.num_flows], F1_large[:, 4 * self.num_flows:5 * self.num_flows]],
                                 dim=1)
        F1_large_b2t = torch.cat(
            [F1_large[:, 2 * self.num_flows:4 * self.num_flows], F1_large[:, 5 * self.num_flows:6 * self.num_flows]],
            dim=1)
        warped_t2b_imgs = multi_warp(x_t2b, F1_large_t2b, mode='forward')
        warped_b2t_imgs = multi_warp(x_b2t, F1_large_b2t, mode='forward')
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, encoding, F1_large), dim=1))
        F2 = (flow0 + flow1)
        F2[:, :4 * self.num_flows] = self._mul_encoding(F2[:, :4 * self.num_flows], encoding_ds)
        F2 = self.conv1(F2)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        F2_large_t2b = torch.cat([F2_large[:, :2 * self.num_flows], F2_large[:, 4 * self.num_flows:5 * self.num_flows]],
                                 dim=1)
        F2_large_b2t = torch.cat(
            [F2_large[:, 2 * self.num_flows:4 * self.num_flows], F2_large[:, 5 * self.num_flows:6 * self.num_flows]],
            dim=1)
        warped_t2b_imgs = multi_warp(x_t2b, F2_large_t2b, mode='forward')
        warped_b2t_imgs = multi_warp(x_b2t, F2_large_b2t, mode='forward')
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, encoding, F2_large), dim=1))
        F3 = (flow0 + flow1 + flow2)
        F3[:, :4 * self.num_flows] = self._mul_encoding(F3[:, :4 * self.num_flows], encoding_ds)
        F3 = self.conv2(F3)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        F3_large_t2b = torch.cat([F3_large[:, :2 * self.num_flows], F3_large[:, 4 * self.num_flows:5 * self.num_flows]],
                                 dim=1)
        F3_large_b2t = torch.cat(
            [F3_large[:, 2 * self.num_flows:4 * self.num_flows], F3_large[:, 5 * self.num_flows:6 * self.num_flows]],
            dim=1)
        warped_t2b_imgs = multi_warp(x_t2b, F3_large_t2b, mode='forward')
        warped_b2t_imgs = multi_warp(x_b2t, F3_large_b2t, mode='forward')
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, encoding, F3_large), dim=1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4[:, :4 * self.num_flows] = self._mul_encoding(F4[:, :4 * self.num_flows], encoding_ds)
        F4 = self.conv3(F4)

        return F4, [F1[:, :4 * self.num_flows],
                    F2[:, :4 * self.num_flows],
                    F3[:, :4 * self.num_flows],
                    F4[:, :4 * self.num_flows]]


if __name__ == '__main__':
    imgs = torch.tensor(np.random.normal(
        0, 1, (4, 6, 256, 256))).float().to(device)
    encodings = torch.tensor(np.random.normal(
        0, 1, (4, 10, 256, 256))).float().to(device)
    flownet = FlowNetMulMulFusion(num_flows=5).cuda()
    flow, _ = flownet(imgs, encodings)
    print(flow.shape)
    # imgs = torch.tensor(np.random.normal(
    #     0, 1, (4, 6, 256, 256))).float().to(device)
    # encodings = torch.tensor(np.random.normal(
    #     0, 1, (4, 6, 256, 256))).float().to(device)
    # flownet = FlowNetLge().cuda()
    # flow, _ = flownet(imgs, encodings)
    # print(flow.shape)
