try:
    from .DualFlowNet import FlowNetMulCatFusion
    from .warplayer import multi_warp
    from .loss import *
except:
    from DualFlowNet import FlowNetMulCatFusion
    from warplayer import multi_warp
    from loss import *
from thop import profile
from data.utils import pad

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )


class ConvDS(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ConvDS, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class WarpedContextNet(nn.Module):
    def __init__(self, c=16, num_flows=3):
        super(WarpedContextNet, self).__init__()
        self.num_flows = num_flows
        self.conv0_0 = ConvDS(3, c)
        self.conv1_0 = ConvDS(c, c)
        self.conv1_1 = conv(num_flows * c, c, kernel_size=1, padding=0, stride=1)
        self.conv2_0 = ConvDS(c, 2 * c)
        self.conv2_1 = conv(num_flows * (2 * c), 2 * c, kernel_size=1, padding=0, stride=1)
        self.conv3_0 = ConvDS(2 * c, 4 * c)
        self.conv3_1 = conv(num_flows * (4 * c), 4 * c, kernel_size=1, padding=0, stride=1)
        self.conv4_0 = ConvDS(4 * c, 8 * c)
        self.conv4_1 = conv(num_flows * (8 * c), 8 * c, kernel_size=1, padding=0, stride=1)

    def forward(self, x, flow):
        x = self.conv0_0(x)
        x = self.conv1_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = multi_warp(x, flow)
        f1 = self.conv1_1(f1)

        x = self.conv2_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = multi_warp(x, flow)
        f2 = self.conv2_1(f2)

        x = self.conv3_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = multi_warp(x, flow)
        f3 = self.conv3_1(f3)

        x = self.conv4_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f4 = multi_warp(x, flow)
        f4 = self.conv4_1(f4)
        return [f1, f2, f3, f4]


class IFEDNet(nn.Module):
    def __init__(self, c=16, num_flows=3):
        self.num_flows = num_flows
        super(IFEDNet, self).__init__()
        self.conv0 = ConvDS(2 * (3 + 2) * num_flows, c)
        self.down0 = ConvDS(c, 2 * c)
        self.down1 = ConvDS(4 * c, 4 * c)  # +2c
        self.down2 = ConvDS(8 * c, 8 * c)  # +4c
        self.down3 = ConvDS(16 * c, 16 * c)  # +8c
        self.up0 = deconv(32 * c, 8 * c)  # +16c
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv1 = deconv(c, 4 * num_flows, 4, 2, 1)

    def forward(self, img_t2b, img_b2t, flow_t2b, flow_b2t, c_t2b, c_b2t):
        warped_t2b_imgs = multi_warp(img_t2b, flow_t2b)  # (n, num_flows*3, h, w)
        warped_b2t_imgs = multi_warp(img_b2t, flow_b2t)  # (n, num_flows*3, h, w)

        d0 = self.conv0(torch.cat((warped_t2b_imgs, warped_b2t_imgs, flow_t2b, flow_b2t), dim=1))
        d0 = self.down0(d0)
        d1 = self.down1(torch.cat((d0, c_t2b[0], c_b2t[0]), dim=1))
        d2 = self.down2(torch.cat((d1, c_t2b[1], c_b2t[1]), dim=1))
        d3 = self.down3(torch.cat((d2, c_t2b[2], c_b2t[2]), dim=1))
        out = self.up0(torch.cat((d3, c_t2b[3], c_b2t[3]), dim=1))
        out = self.up1(torch.cat((out, d2), dim=1))
        out = self.up2(torch.cat((out, d1), dim=1))
        out = self.up3(torch.cat((out, d0), dim=1))
        out = self.conv1(out)

        res = torch.sigmoid(out[:, :3 * self.num_flows]) * 2 - 1
        mask = torch.sigmoid(out[:, 3 * self.num_flows:])  # (n, 3, h, w)
        n, c, h, w = warped_t2b_imgs.shape
        warped_t2b_imgs = warped_t2b_imgs.reshape(n, self.num_flows, 3, h, w)
        warped_b2t_imgs = warped_b2t_imgs.reshape(n, self.num_flows, 3, h, w)
        mask = mask.reshape(n, self.num_flows, 1, h, w)
        warped_imgs = mask * warped_t2b_imgs + (1. - mask) * warped_b2t_imgs
        warped_imgs = warped_imgs.reshape(n, self.num_flows * 3, h, w)
        pred = warped_imgs + res
        pred = torch.clamp(pred, 0, 1)

        return pred


class Criterion:
    def __init__(self, args):
        super(Criterion, self).__init__()
        ratios, losses = loss_parse(args.loss)
        self.losses_name = losses
        self.ratios = ratios
        self.losses = []
        for loss in losses:
            loss_fn = eval('{}(args)'.format(loss))
            self.losses.append(loss_fn)

    def __call__(self, outputs, gts, flows, gt_flows):
        b, c, h, w = outputs.shape
        fb, fc, fh, fw = gt_flows.shape
        losses = {}
        loss_all = None
        for i in range(len(self.losses)):
            if self.losses_name[i].lower().startswith('epe'):
                loss_sub = self.losses[i](flows[0], gt_flows, 1)
                for flow in flows[1:]:
                    loss_sub += self.losses[i](flow, gt_flows, 1)
                loss_sub = self.ratios[i] * loss_sub.mean()
            elif self.losses_name[i].lower().startswith('census'):
                loss_sub = self.ratios[i] * self.losses[i](
                    outputs.reshape(b * int(c // 3), 3, h, w),
                    gts.reshape(b * int(c // 3), 3, h, w)
                ).mean()
            elif self.losses_name[i].lower().startswith('variation'):
                loss_sub = self.losses[i](flows[0].reshape(fb * int(fc // 2), 2, fh, fw), mean=True)
                for flow in flows[1:]:
                    loss_sub += self.losses[i](flow.reshape(fb * int(fc // 2), 2, fh, fw), mean=True)
                loss_sub = self.ratios[i] * loss_sub
            else:
                loss_sub = self.ratios[i] * self.losses[i](outputs, gts)
            losses[self.losses_name[i]] = loss_sub
            if loss_all == None:
                loss_all = loss_sub
            else:
                loss_all += loss_sub
        losses['all'] = loss_all

        return losses


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.flow_net = FlowNetMulCatFusion(num_flows=args.frames)
        self.warped_context_net = WarpedContextNet(c=args.n_feats, num_flows=args.frames)
        self.ife_net = IFEDNet(c=args.n_feats, num_flows=args.frames)
        self.criterion = Criterion(args)

    def forward(self, x, encoding, return_velocity=False):
        _, x_t2b, _, _, x_b2t, _ = torch.chunk(x, chunks=6, dim=1)
        x = torch.cat((x_t2b, x_b2t), dim=1)
        if return_velocity:
            flow, flows, velocity = self.flow_net(x, encoding, return_velocity)
        else:
            flow, flows = self.flow_net(x, encoding)
        flow_t2b, flow_b2t = torch.chunk(flow, chunks=2, dim=1)
        c_t2b = self.warped_context_net(x_t2b, flow_t2b)
        c_b2t = self.warped_context_net(x_b2t, flow_b2t)
        flow_t2b = F.interpolate(flow_t2b, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        flow_b2t = F.interpolate(flow_b2t, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        out = self.ife_net(x_t2b, x_b2t, flow_t2b, flow_b2t, c_t2b, c_b2t)

        if return_velocity:
            return out, flows, velocity
        return out, flows


def inference(model, samples, calc_loss=False, return_velocity=False):
    inputs, gts, gt_flows, dis_encodings = samples

    b, n, c, h, w = inputs.shape  # (8, 6, 3, 256, 256)
    inputs = inputs.reshape(b, n * c, h, w)  # (8, 18, 256, 256)

    b, n, c, h, w = dis_encodings.shape  # (8, 2*ext_frames, 1, 256, 256)
    dis_encodings = dis_encodings.reshape(b, n * c, h, w)  # (8, 2*ext_frames, 256, 256)

    if return_velocity:
        outputs, flows, velocity = model(inputs, dis_encodings, return_velocity)
    else:
        outputs, flows = model(inputs, dis_encodings)
    b, c, h, w = outputs.shape  # (8, 3*ext_frames, 256, 256)
    num_gs = int(c / 3)

    if calc_loss:
        b, num_gs, c, h, w = gts.shape  # (8, ext_frames, 3, 256, 256)
        gts = gts.reshape(b, num_gs * c, h, w).detach()  # (8, 3*ext_frames, 256, 256)

        b, n, c, h, w = gt_flows.shape  # (8, 2*ext_frames, 2, 256, 256)
        gt_flows = gt_flows.reshape(b, n * c, h, w)  # (8, 4*ext_frames, 256, 256)
        gt_flows = (F.interpolate(gt_flows, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5).detach()

        loss = model.criterion(outputs, gts, flows, gt_flows)
        outputs = outputs.reshape(b, num_gs, 3, h, w)
        return outputs, loss
    else:
        outputs = outputs.reshape(b, num_gs, 3, h, w)
        flows = flows[-1]
        flows = F.interpolate(flows, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        flows = flows.reshape(b, 2 * num_gs, 2, h, w)
        if return_velocity:
            velocity = F.interpolate(velocity, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
            velocity = velocity.reshape(b, 2 * num_gs, 2, h, w)
            return outputs, flows, velocity
        return outputs, flows


def cost_profile(model, H, W, seq_length, ext_frames):
    x = torch.randn(1, 6 * seq_length, H, W).cuda()
    x = pad(x)
    encodings = torch.randn(1, 2 * ext_frames, H, W).cuda()
    encodings = pad(encodings)
    flops, params = profile(model, inputs=(x, encodings), verbose=False)

    return flops, params


if __name__ == '__main__':
    from para import Parameter

    args = Parameter().args
    inputs = torch.randn(4, 18, 256, 256).cuda()
    encodings = torch.randn(4, 6, 256, 256).cuda()
    model = Model(args).cuda()
    outputs, flows = model(inputs, encodings)
    print(outputs.shape)
    for flow in flows:
        print(flow.shape)
