from model import Model
from para import Parameter
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
import numpy as np
from data.utils import normalize, normalize_reverse, pad, flow_to_image
from data.distortion_prior import distortion_map

args = Parameter().args
# args.frames = 5
args.frames = 9
# args.frames = 1
args.model = 'DIFE_MulCatFusion'
# args.model = 'DIFE_woprior'

args.return_velocity = True
# args.return_velocity = False

# args.test_checkpoint = '/home/zhong/Desktop/ICCV_Results/dife_ext9_woprior_500/model_best.pth.tar'
# args.test_checkpoint = '/home/zhong/Desktop/ICCV_Results/dife_ext5_woprior_500/model_best.pth.tar'
# args.test_checkpoint = '/home/zhong/Desktop/ICCV_Results/dife_ext5_500/model_best.pth.tar'
# args.test_checkpoint = '/home/zhong/Desktop/ICCV_Results/dife_ext1_500/model_best.pth.tar'

args.test_checkpoint = './checkpoints/iccv_DIFE_MulCatFusion_RS-GOPRO_DS_lmdb_ext9/model_best.pth.tar'
# args.test_checkpoint = './checkpoints/iccv_DIFE_woprior_RS-GOPRO_DS_lmdb_woprior_ext9/model_best.pth.tar'

model = Model(args).cuda()
checkpoint_path = args.test_checkpoint
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'])

ext_frames = args.frames
val_range = 255.
gs_indices = []
if ext_frames == 1:
    gs_indices = [4]
elif ext_frames == 3:
    gs_indices = [0, 4, 8]
elif ext_frames == 5:
    gs_indices = [0, 2, 4, 6, 8]
elif ext_frames == 9:
    gs_indices = list(range(9))

# data_root = '/home/zhong/Dataset/2rs_to_1gs/'
# data_root = '/home/zhong/Desktop/real_demo/'
data_root = './demo/dife/'

# rs_img_t2b_file = join(data_root, 'right_59.png')
# rs_img_b2t_file = join(data_root, 'left_59.png')
# rs_img_t2b_file = join(data_root, 'left_59.png')
# rs_img_b2t_file = join(data_root, 'right_59.png')
# rs_img_b2t_file = join(data_root, 'tr1_2.jpg')
# rs_img_t2b_file = join(data_root, 'tr1_2_2.jpg')

rs_img_t2b_file = join(data_root, 'rs_0.png')
rs_img_b2t_file = join(data_root, 'rs_1.png')

rs_img_t2b = cv2.imread(rs_img_t2b_file).transpose(2, 0, 1)[np.newaxis, ...].repeat(args.test_frames, 0)
rs_img_t2b = torch.from_numpy(rs_img_t2b)
rs_img_b2t = cv2.imread(rs_img_b2t_file).transpose(2, 0, 1)[np.newaxis, ...].repeat(args.test_frames, 0)
rs_img_b2t = torch.from_numpy(rs_img_b2t)
# rs_img_t2b = cv2.imread(rs_img_t2b_file).transpose(2, 0, 1)[np.newaxis, ...].repeat(args.test_frames, 0)
# rs_img_t2b = torch.from_numpy(rs_img_t2b)
# rs_img_b2t = cv2.imread(rs_img_b2t_file).transpose(2, 0, 1)[np.newaxis, ...].repeat(args.test_frames, 0)
# rs_img_b2t = torch.from_numpy(rs_img_b2t)
input_imgs = torch.cat([rs_img_t2b, rs_img_b2t], dim=0).float().cuda()
# input_imgs = F.interpolate(input_imgs, scale_factor=1. / 4, mode="bilinear", align_corners=False,
#                            recompute_scale_factor=False)
input_imgs = input_imgs[None]
H, W = input_imgs.shape[-2:]
input_imgs = pad(input_imgs)

dis_encoding = []
if ext_frames == 1:
    ref_rows = [(H - 1) / 2, ]
else:
    ref_rows = np.linspace(0, H - 1, ext_frames)
for ref_row in ref_rows:
    dis_encoding.append(distortion_map(H, W, ref_row)[np.newaxis, np.newaxis, ...])
for ref_row in ref_rows:
    dis_encoding.append(distortion_map(H, W, ref_row, reverse=True)[np.newaxis, np.newaxis, ...])
dis_encoding = torch.cat(dis_encoding, dim=0)  # (2*ext_frames, 1, h, w)
encoding_imgs = dis_encoding[None].float().cuda()  # (1, 2*ext_frames, 1, h, w)
encoding_imgs = pad(encoding_imgs)

model.eval()
with torch.no_grad():
    input_imgs = normalize(input_imgs, centralize=args.centralize, normalize=args.normalize,
                           val_range=val_range)
    if args.return_velocity:
        pred_imgs, pred_flow_imgs, pred_velocity_imgs = model((input_imgs, None, None, encoding_imgs),
                                                              return_velocity=True)
        pred_velocity_imgs = pred_velocity_imgs.squeeze(dim=0)  # (2*ext_frames, 2, h, w)
        pred_velocities = pred_velocity_imgs.permute(0, 2, 3, 1).cpu().numpy()
    else:
        pred_imgs, pred_flow_imgs = model((input_imgs, None, None, encoding_imgs))
    pred_imgs = pred_imgs.squeeze(dim=0)  # (ext_frames, 3, h, w)
    pred_imgs = normalize_reverse(pred_imgs, centralize=args.centralize, normalize=args.normalize,
                                  val_range=val_range)
    pred_imgs = pred_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
    pred_imgs = np.clip(pred_imgs, 0, val_range).astype(np.uint8)
    pred_flow_imgs = pred_flow_imgs.squeeze(dim=0)  # (2*ext_frames, 2, h, w)
    pred_flows = pred_flow_imgs.permute(0, 2, 3, 1).cpu().numpy()

    foo = 0
    for i, idx in enumerate(gs_indices):
        pred_img_path = join(data_root, '{:08d}_pred_{:03d}.{}'.format(foo, i, 'png'))
        pred_img = pred_imgs[i][:H, :W]
        print(pred_img.shape, pred_img_path)
        cv2.imwrite(pred_img_path, pred_img)
        pred_flow_path = join(data_root, '{:08d}_pred_t2b_flow_{:03d}.{}'.format(foo, i, 'png'))
        pred_flow_img = flow_to_image(pred_flows[i][:H, :W], convert_to_bgr=True)
        cv2.imwrite(pred_flow_path, pred_flow_img)
        pred_flow_path = join(data_root, '{:08d}_pred_b2t_flow_{:03d}.{}'.format(foo, i, 'png'))
        pred_flow = flow_to_image(pred_flows[i + ext_frames][:H, :W], convert_to_bgr=True)
        cv2.imwrite(pred_flow_path, pred_flow)
        # save velocity field
        if args.return_velocity:
            pred_velocity_path = join(data_root, '{:08d}_pred_t2b_velocity_{:03d}.{}'.format(foo, i, 'png'))
            pred_velocity = flow_to_image(pred_velocities[i][:H, :W], convert_to_bgr=True)
            cv2.imwrite(pred_velocity_path, pred_velocity)
            pred_velocity_path = join(data_root, '{:08d}_pred_b2t_velocity_{:03d}.{}'.format(foo, i, 'png'))
            pred_velocity= flow_to_image(pred_velocities[i + ext_frames][:H, :W], convert_to_bgr=True)
            cv2.imwrite(pred_velocity_path, pred_velocity)

# cv2.imshow('t2b', rs_img_t2b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
