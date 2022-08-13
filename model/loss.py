import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MSE(args):
    """
    L2 loss
    """
    return nn.MSELoss()


def L1(args):
    """
    L1 loss
    """
    return nn.L1Loss()


class Charbonnier:
    def __init__(self, args):
        self.epsilon = 1e-6

    def __call__(self, pred, gt):
        return (((pred - gt) ** 2 + self.epsilon) ** 0.5).mean()


def Perceptual(args):
    return PerceptualLoss(loss=nn.L1Loss())


class PerceptualLoss:
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def __call__(self, fake_img, real_img):
        n, c, h, w = fake_img.shape
        fake_img = fake_img.reshape(n * int(c / 3), 3, h, w)
        real_img = real_img.reshape(n * int(c / 3), 3, h, w)
        f_fake = self.contentFunc.forward(fake_img)
        f_real = self.contentFunc.forward(real_img)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class EPE(nn.Module):
    def __init__(self, args):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Census(nn.Module):
    def __init__(self, args):
        super(Census, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, img):
        patches = F.conv2d(img, self.w.to(img.device), padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class GridGradientCentralDiff:
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)

        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d([0, 1, 0, 1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()

        fx_ = torch.tensor([[1, -1], [0, 0]]).cuda()
        fy_ = torch.tensor([[1, 0], [-1, 0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1, 0], [0, -1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_

        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy


class VariationLoss(nn.Module):
    def __init__(self, nc, grad_fn=GridGradientCentralDiff):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)

    def forward(self, image, weight=None, mean=False):
        dx, dy = self.grad_fn(image)
        variation = dx ** 2 + dy ** 2

        if weight is not None:
            variation = variation * weight.float()
            if mean != False:
                return variation.sum() / weight.sum()
        if mean != False:
            return variation.mean()
        return variation.sum()


# Variance loss
def Variation(args):
    return VariationLoss(nc=2)


def loss_parse(loss_str):
    """
    parse loss parameters
    """
    ratios = []
    losses = []
    str_temp = loss_str.split('|')
    for item in str_temp:
        substr_temp = item.split('*')
        ratios.append(float(substr_temp[0]))
        losses.append(substr_temp[1])
    return ratios, losses
