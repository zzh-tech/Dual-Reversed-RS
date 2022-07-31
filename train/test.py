import os
import time
from os.path import join, dirname, isdir

import cv2
import numpy as np
import torch
import torch.nn as nn

from data.utils import normalize, normalize_reverse, pad, flow_to_image, InputPadder
from data.distortion_prior import distortion_map
from model import Model
from .metrics import psnr_calculate, ssim_calculate, lpips_calculate
from .utils import AverageMeter


def test(args, logger):
    """
    test code
    """
    # load model with checkpoint
    if not args.test_only:
        args.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if args.test_save_dir is None:
        args.test_save_dir = logger.save_dir
    model = Model(args).cuda()
    checkpoint_path = args.test_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])

    ds_name = args.dataset
    logger('{} results generating ...'.format(ds_name), prefix='\n')
    if ds_name in ['RS-GOPRO_DS', 'RS-GOPRO_DS_lmdb', 'RS-VFI', 'RS-VFI_lmdb', 'RS-GOPRO_DS_Gap2',
                   'RS-GOPRO_DS_Gap2_lmdb']:
        ds_type = 'test'
        _test_torch(args, logger, model, ds_type)
    else:
        raise NotImplementedError


def _test_torch(args, logger, model, ds_type):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    if args.dataset.startswith('RS-GOPRO'):
        H, W = 540, 960
    elif args.dataset.startswith('RS-VFI'):
        H, W = 512, 512
    padder = InputPadder((H, W))
    val_range = 2.0 ** 8 - 1
    ext_frames = args.frames
    gs_indices = []
    if ext_frames == 1:
        gs_indices = [4]
    elif ext_frames == 3:
        gs_indices = [0, 4, 8]
    elif ext_frames == 5:
        gs_indices = [0, 2, 4, 6, 8]
    elif ext_frames == 9:
        gs_indices = list(range(9))
    dataset_path = join(args.data_root, args.dataset.replace('_lmdb', ''), ds_type)
    seqs = [dir for dir in sorted(os.listdir(dataset_path)) if isdir(join(dataset_path, dir))]

    for seq in seqs:
        logger('seq {} image results generating ...'.format(seq))
        seq_path = join(dataset_path, seq)
        seq_length = int(len(os.listdir(join(seq_path, 'RS'))) // 2)
        dir_name = '_'.join((args.dataset, args.model, ds_type))
        save_dir = join(args.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        suffix = 'png'
        start = 0
        end = args.test_frames  # TODO WARNING: TEST_FRAMES MUST BE 3
        while True:
            for frame_idx in range(start, end):
                if frame_idx != start + 1:
                    continue
                rs_t2b_img_path = join(seq_path, 'RS', '{:08d}_rs_t2b.{}'.format(frame_idx, suffix))
                rs_t2b_img = cv2.imread(rs_t2b_img_path).transpose(2, 0, 1)[np.newaxis, ...].repeat(args.test_frames, 0)
                rs_t2b_img = torch.from_numpy(rs_t2b_img)
                rs_b2t_img_path = join(seq_path, 'RS', '{:08d}_rs_b2t.{}'.format(frame_idx, suffix))
                rs_b2t_img = cv2.imread(rs_b2t_img_path).transpose(2, 0, 1)[np.newaxis, ...].repeat(args.test_frames, 0)
                rs_b2t_img = torch.from_numpy(rs_b2t_img)

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

                gs_imgs = []
                gs_img_paths = [join(seq_path, 'GS', '{:08d}_gs_{:03d}.{}'.format(frame_idx, i, suffix)) for i in
                                gs_indices]
                for gs_img_path in gs_img_paths:
                    gs_imgs.append(cv2.imread(gs_img_path))

                gs_flows = []
                gs_t2b_flows_paths = [join(seq_path, 'FL', '{:08d}_fl_t2b_{:03d}.{}'.format(frame_idx, i, 'npy')) for i
                                      in gs_indices]
                for gs_flow_path in gs_t2b_flows_paths:
                    gs_flows.append(np.load(gs_flow_path))
                gs_b2t_flows_paths = [join(seq_path, 'FL', '{:08d}_fl_b2t_{:03d}.{}'.format(frame_idx, i, 'npy')) for i
                                      in gs_indices]
                for gs_flow_path in gs_b2t_flows_paths:
                    gs_flows.append(np.load(gs_flow_path))

            input_seq = torch.cat((rs_t2b_img, rs_b2t_img), dim=0)[None].float().cuda()  # (1, 2*test_frames, 3, h, w)
            input_seq = pad(input_seq)
            encoding_seq = dis_encoding[None].float().cuda()  # (1, 2*ext_frames, 1, h, w)
            encoding_seq = pad(encoding_seq)
            label_seq = gs_imgs
            flow_seq = gs_flows

            model.eval()
            with torch.no_grad():
                input_seq = normalize(input_seq, centralize=args.centralize, normalize=args.normalize,
                                      val_range=val_range)
                time_start = time.time()
                output_seq, pred_flow_seq = model((input_seq, None, None, encoding_seq))
                output_seq = output_seq.squeeze(dim=0)  # (ext_frames, 3, h, w)
                pred_flow_seq = pred_flow_seq.squeeze(dim=0)  # (2*ext_frames, 2, h, w)
                timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))
            for frame_idx in range(args.past_frames, end - start - args.future_frames):
                # save rs t2b img
                rs_t2b_img = input_seq.squeeze(dim=0)[frame_idx]
                rs_t2b_img = normalize_reverse(rs_t2b_img, centralize=args.centralize, normalize=args.normalize,
                                               val_range=val_range)
                rs_t2b_img = rs_t2b_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                rs_t2b_img = rs_t2b_img.astype(np.uint8)[:H, :W]
                rs_img_path = join(save_dir, '{:08d}_rs_t2b.{}'.format(frame_idx + start, suffix))
                cv2.imwrite(rs_img_path, rs_t2b_img)
                # save rs b2t img
                rs_b2t_img = input_seq.squeeze(dim=0)[frame_idx + args.test_frames]
                rs_b2t_img = normalize_reverse(rs_b2t_img, centralize=args.centralize, normalize=args.normalize,
                                               val_range=val_range)
                rs_b2t_img = rs_b2t_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                rs_b2t_img = rs_b2t_img.astype(np.uint8)[:H, :W]
                rs_img_path = join(save_dir, '{:08d}_rs_b2t.{}'.format(frame_idx + start, suffix))
                cv2.imwrite(rs_img_path, rs_b2t_img)
                # save gs imgs, predicted imgs, gt flows, flows
                gs_imgs = label_seq
                pred_imgs = normalize_reverse(output_seq, centralize=args.centralize, normalize=args.normalize,
                                              val_range=val_range)
                pred_imgs = pred_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
                pred_imgs = np.clip(pred_imgs, 0, val_range).astype(np.uint8)
                gs_flows = flow_seq
                pred_flows = pred_flow_seq.permute(0, 2, 3, 1).cpu().numpy()
                for i, idx in enumerate(gs_indices):
                    gs_img_path = join(save_dir, '{:08d}_gs_{:03d}.{}'.format(frame_idx + start, i, suffix))
                    gs_img = gs_imgs[i]
                    cv2.imwrite(gs_img_path, gs_img)
                    pred_img_path = join(save_dir, '{:08d}_pred_{:03d}.{}'.format(frame_idx + start, i, suffix))
                    pred_img = pred_imgs[i][:H, :W]
                    cv2.imwrite(pred_img_path, pred_img)
                    gs_flow_path = join(save_dir, '{:08d}_gs_t2b_flow_{:03d}.{}'.format(frame_idx + start, i, suffix))
                    gs_flow = flow_to_image(gs_flows[i], convert_to_bgr=True)
                    cv2.imwrite(gs_flow_path, gs_flow)
                    gs_flow_path = join(save_dir, '{:08d}_gs_b2t_flow_{:03d}.{}'.format(frame_idx + start, i, suffix))
                    gs_flow = flow_to_image(gs_flows[i + ext_frames], convert_to_bgr=True)
                    cv2.imwrite(gs_flow_path, gs_flow)
                    pred_flow_path = join(save_dir,
                                          '{:08d}_pred_t2b_flow_{:03d}.{}'.format(frame_idx + start, i, suffix))
                    pred_flow = flow_to_image(pred_flows[i][:H, :W], convert_to_bgr=True)
                    cv2.imwrite(pred_flow_path, pred_flow)
                    pred_flow_path = join(save_dir,
                                          '{:08d}_pred_b2t_flow_{:03d}.{}'.format(frame_idx + start, i, suffix))
                    pred_flow = flow_to_image(pred_flows[i + ext_frames][:H, :W], convert_to_bgr=True)
                    cv2.imwrite(pred_flow_path, pred_flow)

                    if pred_img_path not in results_register:
                        results_register.add(pred_img_path)
                        PSNR.update(psnr_calculate(pred_img, gs_img, val_range=val_range))
                        SSIM.update(ssim_calculate(pred_img, gs_img, val_range=val_range))
                        LPIPS.update(lpips_calculate(pred_img, gs_img))

            if end == seq_length:
                break
            else:
                start = end - args.future_frames - args.past_frames
                end = start + args.test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - args.test_frames

        if args.video:
            fps = 10
            size = (3 * W, 3 * H)
            logger('seq {} video result generating ...'.format(seq))
            path = save_dir
            frame_start = args.past_frames
            frame_end = seq_length - args.future_frames
            file_path = join(dirname(path), '{}.avi'.format(seq))
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video = cv2.VideoWriter(file_path, fourcc, fps, size)
            for i in range(frame_start, frame_end):
                for j in range(ext_frames):
                    frame = []
                    # first row: [diff, pred_img, gs_img]
                    imgs = []
                    # pred img
                    img_path = join(path, '{:08d}_pred_{:03d}.{}'.format(i, j, suffix))
                    pred_img = cv2.imread(img_path)
                    pred_img_mk = cv2.putText(pred_img.copy(), 'pred {} {}'.format(i, j),
                                              (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert pred_img.shape == (H, W, 3), pred_img.shape
                    # gs img
                    img_path = join(path, '{:08d}_gs_{:03d}.{}'.format(i, j, suffix))
                    gs_img = cv2.imread(img_path)
                    gs_img_mk = cv2.putText(gs_img.copy(), 'gs {} {}'.format(i, j),
                                            (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert gs_img.shape == (H, W, 3), gs_img.shape
                    # pred diff
                    pred_diff = cv2.absdiff(gs_img, pred_img)
                    pred_diff = cv2.cvtColor(pred_diff, cv2.COLOR_BGR2GRAY)
                    pred_diff = cv2.cvtColor(pred_diff, cv2.COLOR_GRAY2BGR)
                    pred_diff = cv2.putText(pred_diff, 'diff {} {}'.format(i, j),
                                            (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert pred_diff.shape == (H, W, 3), pred_diff.shape
                    imgs.append(pred_diff)
                    imgs.append(pred_img_mk)
                    imgs.append(gs_img_mk)

                    imgs = np.concatenate(imgs, axis=1)
                    assert imgs.shape == (H, 3 * W, 3), imgs.shape
                    frame.append(imgs)

                    # second row: [rs_t2b_img, pred_t2b_flow, gs_t2b_flow]
                    imgs = []
                    # rs t2b img
                    img_path = join(path, '{:08d}_rs_t2b.{}'.format(i, suffix))
                    rs_t2b_img = cv2.imread(img_path)
                    rs_t2b_img = cv2.putText(rs_t2b_img, 'rs t2b {}'.format(i), (60, 60),
                                             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert rs_t2b_img.shape == (H, W, 3), rs_t2b_img.shape
                    # pred t2b flow
                    img_path = join(path, '{:08d}_pred_t2b_flow_{:03d}.{}'.format(i, j, suffix))
                    pred_flow = cv2.imread(img_path)
                    pred_flow = cv2.putText(pred_flow, 'pred t2b fl {} {}'.format(i, j),
                                            (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert pred_flow.shape == (H, W, 3), pred_flow.shape
                    # gs t2b flow
                    img_path = join(path, '{:08d}_gs_t2b_flow_{:03d}.{}'.format(i, j, suffix))
                    gs_flow = cv2.imread(img_path)
                    # gs_flow = padder.unpad(gs_flow)
                    gs_flow = cv2.putText(gs_flow, 'gs t2b fl {} {}'.format(i, j),
                                          (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert gs_flow.shape == (H, W, 3), gs_flow.shape
                    imgs.append(rs_t2b_img)
                    imgs.append(pred_flow)
                    imgs.append(gs_flow)

                    imgs = np.concatenate(imgs, axis=1)
                    assert imgs.shape == (H, 3 * W, 3), imgs.shape
                    frame.append(imgs)

                    # third row: [rs_b2t_img, pred_b2t_flow, gs_b2t_flow]
                    imgs = []
                    # rs t=b2t img
                    img_path = join(path, '{:08d}_rs_b2t.{}'.format(i, suffix))
                    rs_b2t_img = cv2.imread(img_path)
                    rs_b2t_img = cv2.putText(rs_b2t_img, 'rs b2t {}'.format(i), (60, 60),
                                             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert rs_b2t_img.shape == (H, W, 3), rs_b2t_img.shape
                    # pred b2t flow
                    img_path = join(path, '{:08d}_pred_b2t_flow_{:03d}.{}'.format(i, j, suffix))
                    pred_flow = cv2.imread(img_path)
                    pred_flow = cv2.putText(pred_flow, 'pred b2t fl {} {}'.format(i, j),
                                            (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert pred_flow.shape == (H, W, 3), pred_flow.shape
                    # gs b2t flow
                    img_path = join(path, '{:08d}_gs_b2t_flow_{:03d}.{}'.format(i, j, suffix))
                    gs_flow = cv2.imread(img_path)
                    # gs_flow = padder.unpad(gs_flow)
                    gs_flow = cv2.putText(gs_flow, 'gs b2t fl {} {}'.format(i, j),
                                          (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    assert gs_flow.shape == (H, W, 3), gs_flow.shape
                    imgs.append(rs_b2t_img)
                    imgs.append(pred_flow)
                    imgs.append(gs_flow)

                    imgs = np.concatenate(imgs, axis=1)
                    assert imgs.shape == (H, 3 * W, 3), imgs.shape
                    frame.append(imgs)

                    frame = np.concatenate(frame, axis=0)
                    video.write(frame)
            video.release()

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Test LPIPS : {}'.format(LPIPS.avg))
    logger('Average time per image: {}'.format(timer.avg))
