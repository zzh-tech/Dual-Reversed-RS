import torch
import torch.nn.functional as F

import model.softsplat as softsplat

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def multi_warp(img, flows, device='cuda', mode='backward'):
    if mode == 'backward':
        warped_imgs = multi_backward_warp(img, flows, device)
    elif mode == 'forward':
        warped_imgs = multi_forward_warp(img, flows)
    return warped_imgs


# TODO: NO LOOP
def multi_backward_warp(img, flows, device='cuda'):
    num_flows = int(flows.shape[1] // 2)
    warped_imgs = []
    for i in range(num_flows):
        warped_imgs.append(warp(img, flows[:, 2 * i:2 * (i + 1)], device))
    warped_imgs = torch.cat(warped_imgs, dim=1)
    return warped_imgs

def multi_forward_warp(img, flows):
    num_flows = int(flows.shape[1] // 3)
    warped_imgs = []
    metrics = flows[:, -num_flows:]
    flows = flows[:, :-num_flows]
    for i in range(num_flows):
        flow = flows[:, 2 * i: 2 * (i + 1)]
        metric = metrics[:, i: i + 1]
        warped_img = softsplat.FunctionSoftsplat(tenInput=img, tenFlow=flow, tenMetric=-20.0 * F.sigmoid(metric),
                                                 strType='softmax')
        # warped_img = softsplat.FunctionSoftsplat(tenInput=img, tenFlow=flow, tenMetric=None, strType='average')
        warped_imgs.append(warped_img)
    warped_imgs = torch.cat(warped_imgs, dim=1)
    return warped_imgs
