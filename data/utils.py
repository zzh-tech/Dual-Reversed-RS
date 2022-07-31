import os
import cv2
import numpy as np
from os.path import dirname, basename, join
import torch.nn.functional as F


def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x


def pad(img, ratio=32):
    if len(img.shape) == 5:
        b, n, c, h, w = img.shape
        img = img.reshape(b * n, c, h, w)
        ph = ((h - 1) // ratio + 1) * ratio
        pw = ((w - 1) // ratio + 1) * ratio
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding, mode='replicate')
        img = img.reshape(b, n, c, ph, pw)
        return img
    elif len(img.shape) == 4:
        n, c, h, w = img.shape
        ph = ((h - 1) // ratio + 1) * ratio
        pw = ((w - 1) // ratio + 1) * ratio
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding, mode='replicate')
        return img


def adjust_gamma(image, gamma=1.0):
    image = image.astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** inv_gamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


class VideoEdit:
    """
    Tool for video edit
    """

    @staticmethod
    def imgs2video(src_path, dst_path, fps):
        """
        Parameters:
            src_path: path of image directory
            dst_path: path of generated video
            fps: frame rate for generated video
        """
        assert os.path.exists(src_path)
        img_files = os.listdir(src_path)
        imgs_files = sorted(img_files, key=lambda x: int(x.split('.')[0]))
        os.makedirs(dirname(dst_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        img = cv2.imread(join(src_path, imgs_files[0]))
        H, W, _ = img.shape
        size = (W, H)
        video = cv2.VideoWriter(dst_path, fourcc, fps, size)
        video.write(img)
        for img_file in imgs_files[1:]:
            img = cv2.imread(join(src_path, img_file))
            video.write(img)
        video.release()

    def __init__(self, video_path, verbose=False):
        """ Initialize VideoEdit

        Parameters:
            video_path: path of the video file
            verbose: whether to show logging
        """
        self.vidcap = cv2.VideoCapture(video_path)
        self.success, self.img = self.vidcap.read()
        self.verbose = verbose
        if not self.success:
            raise IndexError('Cannot read a frame from the video at {}.'.format(video_path))
        assert len(self.img.shape) == 3
        self.H, self.W, self.C = self.img.shape
        if self.verbose:
            print('frame size of input video: ({}, {}, {})'.format(self.H, self.W, self.C))

    def gen_rs_video(self, rs_video_path, src_fps, dst_fps=30, disp_fps=15, row_syn_frames=10, row_gap_frames=1,
                     gt_frames=9, scan_mode='t2b', gamma=False):
        """ Synthesize a video with rolling shutter distortion and corresponding image pairs

        Parameters:
            rs_video_path: path to save the rs video
            src_fps: frame rate of the input gs video
            dst_fps: frame rate of the generated rs video
            disp_fps: frame rate of the generated rs video for display
            row_syn_frames: number of frames used to synthesize one row of rs frame, i.e., exposure time for each row
            row_gap_frames: number of frames gaped between adjacent rows, i.e., readout time for each row
            gt_frames: number of gt images for one rs image
            scan_mode: "t2b | b2t | both"; scanning mode for rs video generation;
                        default way is from top to bottom
            gamma: consider gamma correction
        """
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if scan_mode == 't2b' or scan_mode == 'b2t':
            size = (2 * self.W, self.H)
        elif scan_mode == 'both':
            size = (3 * self.W, self.H)
        else:
            raise ValueError('should be one of [t2b | b2t | both]')
        video = cv2.VideoWriter(rs_video_path, fourcc, disp_fps, size)
        img_dir = join(dirname(rs_video_path), basename(rs_video_path).split('.')[0])
        img_gt_dir = join(img_dir, 'GS')
        img_rs_dir = join(img_dir, 'RS')
        os.makedirs(img_gt_dir, exist_ok=True)
        os.makedirs(img_rs_dir, exist_ok=True)
        temp_imgs = [self.img, ]
        count = 0
        num_temp_imgs = int(src_fps / dst_fps)
        num_valid_temp_imgs = (self.H - 1) * row_gap_frames + row_syn_frames
        assert num_temp_imgs >= num_valid_temp_imgs
        if self.verbose:
            print('consider gamma correction: {}'.format(gamma))
            print('{} gs images per rs image'.format(num_temp_imgs))
            print('duty cycle: {:.4f}'.format(num_valid_temp_imgs / num_temp_imgs))
            print('exposure time {:.4f} ms'.format(row_syn_frames * (1 / src_fps) * (10 ** 3)))
            print('readout time {:.4f} ms'.format(row_gap_frames * (1 / src_fps) * (10 ** 3)))
        while self.success:
            if len(temp_imgs) < num_temp_imgs:
                self.success, self.img = self.vidcap.read()
                temp_imgs.append(self.img)
            elif len(temp_imgs) == num_temp_imgs:
                temp_imgs = temp_imgs[:num_valid_temp_imgs]
                # generate gt imgs
                gs_imgs = self._generate_gt_imgs(temp_imgs, gamma, gt_frames, row_syn_frames)
                assert len(gs_imgs) == gt_frames
                for i, gs_img in enumerate(gs_imgs):
                    cv2.imwrite(join(img_gt_dir, '{:08d}_gs_{:03d}.png'.format(count, i)), gs_img)
                gs_img = gs_imgs[gt_frames // 2]
                # generate rs img
                if scan_mode == 'both':
                    t2b_rs_img = self._generate_rs_img(temp_imgs, gamma, row_syn_frames, row_gap_frames, 't2b')
                    b2t_rs_img = self._generate_rs_img(temp_imgs, gamma, row_syn_frames, row_gap_frames, 'b2t')
                    cv2.imwrite(join(img_rs_dir, '{:08d}_rs_t2b.png'.format(count)), t2b_rs_img)
                    cv2.imwrite(join(img_rs_dir, '{:08d}_rs_b2t.png'.format(count)), b2t_rs_img)
                    img = np.concatenate((t2b_rs_img, b2t_rs_img, gs_img), axis=1).astype(np.uint8)
                else:
                    rs_img = self._generate_rs_img(temp_imgs, gamma, row_syn_frames, row_gap_frames, scan_mode)
                    cv2.imwrite(join(img_rs_dir, '{:08d}_rs_{}.png'.format(count, scan_mode)), rs_img)
                    img = np.concatenate((rs_img, gs_img), axis=1).astype(np.uint8)
                video.write(img)
                count += 1
                temp_imgs = []
        video.release()
        if self.verbose:
            print('create {}'.format(rs_video_path))

    def _generate_gt_imgs(self, imgs: list, gamma: bool, gt_frames: int, syn_frames=10) -> list:
        """ Synthesize multiple gt images from a list of gs images

        Parameter:
            imgs: a list of global shutter images
            gamma: consider gamma correction
            gt_frames: number of gt images
            syn_frames: number of frames used to synthesize one gt frame, i.e., exposure time for gt image
        Returns:
            gt_imgs: synthesized gt images
        """
        gt_imgs = []
        num_imgs = len(imgs)
        assert num_imgs >= gt_frames * syn_frames
        gt_indices = np.linspace(start=0, stop=num_imgs - syn_frames, num=gt_frames, dtype=np.int)
        for i in gt_indices:
            if gamma:
                temp_imgs = [adjust_gamma(img, gamma=1 / 2.2) for img in imgs[i:i + syn_frames]]
                img = adjust_gamma(np.mean(temp_imgs, axis=0), gamma=2.2)
            else:
                img = np.mean(imgs[i:i + syn_frames], axis=0)
            gt_imgs.append(img)
        return gt_imgs

    def _generate_rs_img(self, imgs: list, gamma: bool, row_syn_frames=10, row_gap_frames=1,
                         scan_mode='t2b') -> np.array:
        """ Synthesize a rs image from a list of gs images

        Parameters:
            imgs: a list of global shutter images
            gamma: consider gamma correction
            row_syn_frames: number of frames used to synthesize one row of rs frame, i.e., exposure time for each row
            row_gap_frames: number of frames gaped between adjacent rows, i.e., readout time for each row
            scan_mode: "t2b | b2t"; scanning mode for rs video generation;
                    default way is from top to bottom
        Returns:
            rs_img: synthesized image with rolling shutter distortion
        """
        assert len(imgs) == (self.H - 1) * row_gap_frames + row_syn_frames, \
            'number of gs frames should be {} but given {}'.format(
                (self.H - 1) * row_gap_frames + row_syn_frames, len(imgs)
            )
        rs_rows = []
        for r in range(self.H):
            if scan_mode == 't2b':
                if gamma:
                    img_row = adjust_gamma(np.mean(
                        [adjust_gamma(img[r], gamma=1 / 2.2) for img in
                         imgs[r * row_gap_frames:r * row_gap_frames + row_syn_frames]],
                        axis=0), gamma=2.2)
                else:
                    img_row = np.mean(
                        [img[r] for img in imgs[r * row_gap_frames:r * row_gap_frames + row_syn_frames]],
                        axis=0)
                rs_rows.append(img_row)
            elif scan_mode == 'b2t':
                if gamma:
                    img_row = adjust_gamma(np.mean(
                        [adjust_gamma(img[self.H - r - 1], gamma=1 / 2.2) for img in
                         imgs[r * row_gap_frames:r * row_gap_frames + row_syn_frames]],
                        axis=0), gamma=2.2)
                else:
                    img_row = np.mean(
                        [img[self.H - r - 1] for img in imgs[r * row_gap_frames:r * row_gap_frames + row_syn_frames]],
                        axis=0)
                rs_rows.insert(0, img_row)
        rs_img = np.stack(rs_rows, axis=0)
        return rs_img


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[:2]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[c[0]:c[1], c[2]:c[3]]
