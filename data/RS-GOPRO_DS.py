"""
RS-GOPRO dataset
Synthesized rolling shutter distortion using high-fps video (GOPRO-Large)
"""

import os
import random
from os.path import join
from time import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .utils import normalize, flow_to_image
    from .distortion_prior import distortion_map
except:
    from utils import normalize, flow_to_image
    from distortion_prior import distortion_map
from model.warplayer import warp, multi_warp


class RSGOPRO(Dataset):
    """ Dataset class for RS-GOPRO"""

    def __init__(self, path, future_frames=1, past_frames=1, ext_frames=9, crop_size=(256, 256), centralize=False,
                 normalize=True, flow=False):
        """
        Initialize dataset class

        Paramters:
            path: where GOPRO-RS locates
            future_frames: number of frames in the future
            past_frames: number of frames in the past
            ext_frames: number of extracted ground truth frames for single rs frame, [1 | 3 | 5| 9]
            crop_size: patch size of random cropping
            centralize: subtract half value range
            normalize: divide value range
        """
        super(RSGOPRO, self).__init__()
        self.H = 540
        self.W = 960
        self.flow = flow
        assert ext_frames in [1, 3, 5, 9], "ext_frames should be [1 | 3 | 5 | 9]"
        self.ext_frames = ext_frames
        self.dis_encoding = self._gen_dis_encoding(self.H, self.W, ext_frames)
        assert self.dis_encoding.shape == (2 * ext_frames, self.H, self.W, 1), self.dis_encoding.shape
        self.normalize = normalize
        self.centralize = centralize
        self.num_ff = future_frames
        self.num_pf = past_frames
        self._samples = self._generate_samples(path)
        self.crop_h, self.crop_w = crop_size

    def _generate_samples(self, path):
        samples = []
        seqs = [seq for seq in sorted(os.listdir(path)) if not seq.endswith('avi')]
        for seq in seqs:
            seq_path = join(path, seq)
            seq_num = len(os.listdir(join(seq_path, 'RS'))) // 2
            for i in range(self.num_pf, seq_num - self.num_ff):
                sample = dict()
                sample['RS'] = {}
                sample['RS']['t2b'] = [join(seq_path, 'RS', '{:08d}_rs_t2b.png'.format(j)) for j in
                                       range(i - self.num_pf, i + self.num_ff + 1)]
                sample['RS']['b2t'] = [join(seq_path, 'RS', '{:08d}_rs_b2t.png'.format(j)) for j in
                                       range(i - self.num_pf, i + self.num_ff + 1)]
                if self.ext_frames == 1:
                    gs_indices = [4]
                elif self.ext_frames == 3:
                    gs_indices = [0, 4, 8]
                elif self.ext_frames == 5:
                    gs_indices = [0, 2, 4, 6, 8]
                elif self.ext_frames == 9:
                    gs_indices = list(range(9))
                else:
                    raise NotImplementedError
                sample['GS'] = [join(seq_path, 'GS', '{:08d}_gs_{:03d}.png').format(i, j) for j in gs_indices]
                sample['FL'] = {}
                sample['FL']['t2b'] = [join(seq_path, 'FL', '{:08d}_fl_t2b_{:03d}.npy').format(i, j) for j in
                                       gs_indices]
                sample['FL']['b2t'] = [join(seq_path, 'FL', '{:08d}_fl_b2t_{:03d}.npy').format(i, j) for j in
                                       gs_indices]
                samples.append(sample)
        return samples

    def _gen_dis_encoding(self, h, w, ext_frames):
        dis_encoding = []
        if ext_frames == 1:
            ref_rows = [(h - 1) / 2, ]
        else:
            ref_rows = np.linspace(0, h - 1, ext_frames)
        for ref_row in ref_rows:
            dis_encoding.append(distortion_map(h, w, ref_row)[..., np.newaxis])
        for ref_row in ref_rows:
            dis_encoding.append(distortion_map(h, w, ref_row, reverse=True)[..., np.newaxis])
        dis_encoding = np.stack(dis_encoding, axis=0)  # (2*ext_frames, h, w, 1)
        return dis_encoding

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        rs_imgs, gs_imgs, fl_imgs, prior_imgs = self._load_sample(self._samples[idx])
        assert rs_imgs.shape == (2 * (self.num_ff + 1 + self.num_pf), 3, self.crop_h, self.crop_w), rs_imgs.shape
        assert gs_imgs.shape == (self.ext_frames, 3, self.crop_h, self.crop_w), gs_imgs.shape
        assert fl_imgs.shape == (2 * self.ext_frames, 2, self.crop_h, self.crop_w), fl_imgs.shape
        assert prior_imgs.shape == (2 * self.ext_frames, 1, self.crop_h, self.crop_w), prior_imgs.shape
        rs_imgs = normalize(rs_imgs, normalize=self.normalize, centralize=self.centralize)
        gs_imgs = normalize(gs_imgs, normalize=self.normalize, centralize=self.centralize)
        return rs_imgs, gs_imgs, fl_imgs, prior_imgs

    def _load_sample(self, sample):
        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        rs_imgs, gs_imgs, fl_imgs, prior_imgs = [], [], [], []
        for rs_img_path in sample['RS']['t2b']:
            img = self._data_augmentation(cv2.imread(rs_img_path), top, left)
            rs_imgs.append(img)
        for rs_img_path in sample['RS']['b2t']:
            img = self._data_augmentation(cv2.imread(rs_img_path), top, left)
            rs_imgs.append(img)
        for gs_img_path in sample['GS']:
            img = self._data_augmentation(cv2.imread(gs_img_path), top, left)
            gs_imgs.append(img)
        for fl_img_path in sample['FL']['t2b']:
            if self.flow:
                img = np.load(fl_img_path)
            else:
                img = np.zeros((self.H, self.W, 2), dtype='float32')
            img = self._data_augmentation(img, top, left)
            fl_imgs.append(img)
        for fl_img_path in sample['FL']['b2t']:
            if self.flow:
                img = np.load(fl_img_path)
            else:
                img = np.zeros((self.H, self.W, 2), dtype='float32')
            img = self._data_augmentation(img, top, left)
            fl_imgs.append(img)
        for img in self.dis_encoding:
            img = self._data_augmentation(img.copy(), top, left)
            prior_imgs.append(img)
        rs_imgs = torch.stack(rs_imgs, dim=0)
        gs_imgs = torch.stack(gs_imgs, dim=0)
        fl_imgs = torch.stack(fl_imgs, dim=0)
        prior_imgs = torch.stack(prior_imgs, dim=0)
        return rs_imgs, gs_imgs, fl_imgs, prior_imgs

    def _data_augmentation(self, img, top, left):
        img = img[top: top + self.crop_h, left: left + self.crop_w]
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).float()
        return img

    def visualize(self, idx):
        rs_imgs, gs_imgs, fl_imgs, prior_imgs = self[idx]
        rs_frames = rs_imgs.shape[0] // 2
        gs_frames = gs_imgs.shape[0]
        # show imgs for t2b mode
        imgs = []
        for i in range(rs_frames):
            img = rs_imgs[i].permute(1, 2, 0).numpy()
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('rs t2b', imgs)
        # show imgs for b2t mode
        imgs, fimgs = [], []
        for i in range(rs_frames, int(2 * rs_frames)):
            img = rs_imgs[i].permute(1, 2, 0).numpy()
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('rs b2t', imgs)
        # show gs imgs and flows
        imgs, t2b_flows, b2t_flows = [], [], []
        for i in range(gs_frames):
            img = gs_imgs[i].permute(1, 2, 0).numpy()
            t2b_flow = normalize(flow_to_image(fl_imgs[i].permute(1, 2, 0).numpy(), convert_to_bgr=True),
                                 normalize=self.normalize, centralize=self.centralize)
            b2t_flow = normalize(
                flow_to_image(fl_imgs[gs_imgs.shape[0] + i].permute(1, 2, 0).numpy(), convert_to_bgr=True),
                normalize=self.normalize, centralize=self.centralize)
            imgs.append(img)
            t2b_flows.append(t2b_flow)
            b2t_flows.append(b2t_flow)
        imgs = np.concatenate(imgs, axis=1)
        t2b_flows = np.concatenate(t2b_flows, axis=1)
        b2t_flows = np.concatenate(b2t_flows, axis=1)
        cv2.imshow('gs, t2b flows, b2t flows', np.concatenate((imgs, t2b_flows, b2t_flows), axis=0))
        # warped imgs according to t2b flows
        t2b_imgs = rs_imgs[rs_frames // 2].repeat(gs_frames, 1, 1, 1)
        t2b_flows = fl_imgs[:gs_frames]
        warped_imgs = warp(t2b_imgs, t2b_flows, device='cpu')
        n, c, h, w = warped_imgs.shape
        warped_imgs = warped_imgs.permute(2, 0, 3, 1).reshape(h, n * w, c).numpy()
        cv2.imshow('warped t2b rs', warped_imgs)
        # warped imgs according to b2t flows
        b2t_imgs = rs_imgs[rs_frames + rs_frames // 2].repeat(gs_frames, 1, 1, 1)
        b2t_flows = fl_imgs[gs_frames:]
        warped_imgs = warp(b2t_imgs, b2t_flows, device='cpu')
        n, c, h, w = warped_imgs.shape
        warped_imgs = warped_imgs.permute(2, 0, 3, 1).reshape(h, n * w, c).numpy()
        cv2.imshow('warped b2t rs', warped_imgs)
        # show distortion t2b prior maps
        imgs = []
        for img in prior_imgs[:gs_frames]:
            img = (img[0].numpy() + 1) * (255. / 2)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('t2b prior maps', imgs)
        # show distortion b2t prior maps
        imgs = []
        for img in prior_imgs[gs_frames:]:
            img = (img[0].numpy() + 1) * (255. / 2)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('b2t prior maps', imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Dataloader:
    def __init__(self, para, device_id=0, ds_type='train'):
        path = join(para.data_root, para.dataset, ds_type)
        flow = True
        if not 'EPE' in para.loss:
            flow = False
        dataset = RSGOPRO(path, para.future_frames, para.past_frames, para.frames, para.patch_size, para.centralize,
                          para.normalize, flow)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler,
                drop_last=True
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True,
                drop_last=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len


if __name__ == '__main__':
    # path = '/home/zhong/Dataset/RS-GOPRO_DS/test/'
    # ds = RSGOPRO(path=path, crop_size=(256, 256), ext_frames=5)
    # idx = random.randint(0, len(ds) - 1)
    # ds.visualize(idx)
    import sys

    sys.path.append('..')
    from para import Parameter

    args = Parameter().args
    args.dataset = 'RS-GOPRO_DS'
    args.frames = 5
    args.patch_size = [256, 256]
    dl = Dataloader(args)
    count = 10
    start_time = time()
    for x, y, z, p in dl:
        print(x.shape, y.shape, z.shape, p.shape)
        count -= 1
        if count == 0:
            break
    print(time() - start_time)
