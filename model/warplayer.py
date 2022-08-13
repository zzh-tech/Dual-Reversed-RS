import torch
import torch.nn.functional as F

backwarp_tenGrid = {}


def warp(tenInput, tenFlow, device='cuda'):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=torch.clamp(g, -1, 1), mode='bilinear',
                         padding_mode='zeros', align_corners=True)


def multi_warp(img, flows, device='cuda'):
    return multi_backward_warp(img, flows, device)


def multi_backward_warp(img, flows, device='cuda'):
    num_flows = int(flows.shape[1] // 2)
    warped_imgs = []
    for i in range(num_flows):
        warped_imgs.append(warp(img, flows[:, 2 * i:2 * (i + 1)], device))
    warped_imgs = torch.cat(warped_imgs, dim=1)
    return warped_imgs
