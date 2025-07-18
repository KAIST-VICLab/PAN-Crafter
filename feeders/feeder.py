# --------------------------------------------------------
# PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening
# Copyright (c) 2025 Jeonghyeok Do, Sungpyo Kim, Geunhyuk Youk, Jaehyup Lee†, and Munchurl Kim†
#
# This code is released under the MIT License (see LICENSE file for details).
#
# This software is licensed for **non-commercial research and educational use only**.
# For commercial use, please contact: mkimee@kaist.ac.kr
# --------------------------------------------------------

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import h5py
import random

class PanFeeder(Dataset):
    def __init__(self, dataroot, max_pixel=0., crop=False, hflip=False, vflip=False, rot=False, crop_ratio=0.5, ms_size=16):
        random.seed(2025)
        self.dataroot = dataroot
        if max_pixel == 0.:
            if "wv3" in dataroot or "qb" in dataroot or "wv2" in dataroot:
                self.max_pixel = 2047.
            elif "gf2" in dataroot:
                self.max_pixel = 1023.
            else:
                self.max_pixel = 1.
                print("Unsupported dataset.")

        if 'train' in self.dataroot:
            trainroot = self.dataroot
            trainroot_pan = trainroot.replace(".h5", "_pan.h5")

            with h5py.File(trainroot, 'r') as h5_file:
                if "/gt" in h5_file:
                    self.has_gt = True
                    train_gt = h5_file['gt'][:].transpose(0, 2, 3, 1)
                else:
                    self.has_gt = False

                train_lms = h5_file['lms'][:].transpose(0, 2, 3, 1)
                train_ms = h5_file['ms'][:].transpose(0, 2, 3, 1)
                train_pan = h5_file['pan'][:].transpose(0, 2, 3, 1)

            with h5py.File(trainroot_pan, 'r') as h5_file:
                train_lpan = h5_file['lpan'][:].transpose(0, 2, 3, 1)

            if self.has_gt:
                self.gt = train_gt

            self.lms = train_lms
            self.ms = train_ms
            self.pan = train_pan
            self.lpan = train_lpan

        else:
            with h5py.File(self.dataroot, 'r') as h5_file:
                if "/gt" in h5_file:
                    self.has_gt = True
                    self.gt = h5_file['gt'][:].transpose(0, 2, 3, 1)
                else:
                    self.has_gt = False

                self.lms = h5_file['lms'][:].transpose(0, 2, 3, 1)
                self.ms = h5_file['ms'][:].transpose(0, 2, 3, 1)
                self.pan = h5_file['pan'][:].transpose(0, 2, 3, 1)

            with h5py.File(self.dataroot.replace(".h5", "_pan.h5"), 'r') as h5_file:
                self.lpan = h5_file['lpan'][:].transpose(0, 2, 3, 1)

        self.crop = crop
        self.hflip = hflip
        self.vflip = vflip
        self.rot = rot
        self.crop_ratio = crop_ratio
        self.ms_size = ms_size

        if 'train' in self.dataroot:
            self.split = 'train'
        elif 'valid' in self.dataroot:
            self.split = 'val'
        elif 'reduced' in self.dataroot:
            self.split = 'test_reduced'
        elif 'full' in self.dataroot:
            self.split = 'test_full'
        else:
            print("Unsupported file format.")

    def augment(self, gt, lms, ms, lpan, pan):
        # random crop
        if self.crop:
            ratio = (1 - self.crop_ratio) * random.random() + self.crop_ratio
            ms_p = np.round(self.ms_size * ratio).astype(int)
            pan_p = 4 * ms_p
            ms_x = random.randrange(0, self.ms_size - ms_p + 1)
            ms_y = random.randrange(0, self.ms_size - ms_p + 1)
            (pan_x, pan_y) = (4 * ms_x, 4 * ms_y)
            gt = gt[pan_y:pan_y + pan_p, pan_x:pan_x + pan_p, :]
            lms = lms[pan_y:pan_y + pan_p, pan_x:pan_x + pan_p, :]
            ms = ms[ms_y:ms_y + ms_p, ms_x:ms_x + ms_p, :]
            lpan = lpan[ms_y:ms_y + ms_p, ms_x:ms_x + ms_p, :]
            pan = pan[pan_y:pan_y + pan_p, pan_x:pan_x + pan_p, :]
            gt = cv2.resize(gt, (4 * self.ms_size, 4 * self.ms_size), interpolation=cv2.INTER_CUBIC)
            lms = cv2.resize(lms, (4 * self.ms_size, 4 * self.ms_size), interpolation=cv2.INTER_CUBIC)
            ms = cv2.resize(ms, (self.ms_size, self.ms_size), interpolation=cv2.INTER_CUBIC)
            lpan = cv2.resize(lpan, (self.ms_size, self.ms_size), interpolation=cv2.INTER_CUBIC)
            pan = cv2.resize(pan, (4 * self.ms_size, 4 * self.ms_size), interpolation=cv2.INTER_CUBIC)
            lpan = lpan[..., np.newaxis]
            pan = pan[..., np.newaxis]

        # random horizontal flip
        if self.hflip:
            gt = gt[:, ::-1, :]
            lms = lms[:, ::-1, :]
            ms = ms[:, ::-1, :]
            lpan = lpan[:, ::-1, :]
            pan = pan[:, ::-1, :]

        # random vertical flip
        if self.vflip:
            gt = gt[::-1, :, :]
            lms = lms[::-1, :, :]
            ms = ms[::-1, :, :]
            lpan = lpan[::-1, :, :]
            pan = pan[::-1, :, :]

        # random rotate
        if self.rot:
            rot = random.randint(0, 3)
            gt = np.rot90(gt, rot, (0, 1))
            lms = np.rot90(lms, rot, (0, 1))
            ms = np.rot90(ms, rot, (0, 1))
            lpan = np.rot90(lpan, rot, (0, 1))
            pan = np.rot90(pan, rot, (0, 1))

        return gt, lms, ms, lpan, pan

    def augment_without_gt(self, lms, ms, lpan, pan):
        # random crop
        if self.crop:
            ratio = (1 - self.crop_ratio) * random.random() + self.crop_ratio
            ms_p = np.round(self.ms_size * ratio).astype(int)
            pan_p = 4 * ms_p
            ms_x = random.randrange(0, self.ms_size - ms_p + 1)
            ms_y = random.randrange(0, self.ms_size - ms_p + 1)
            (pan_x, pan_y) = (4 * ms_x, 4 * ms_y)
            lms = lms[pan_y:pan_y + pan_p, pan_x:pan_x + pan_p, :]
            ms = ms[ms_y:ms_y + ms_p, ms_x:ms_x + ms_p, :]
            pan = pan[pan_y:pan_y + pan_p, pan_x:pan_x + pan_p, :]
            lms = cv2.resize(lms, (4 * self.ms_size, 4 * self.ms_size), interpolation=cv2.INTER_CUBIC)
            ms = cv2.resize(ms, (self.ms_size, self.ms_size), interpolation=cv2.INTER_CUBIC)
            lpan = cv2.resize(lpan, (self.ms_size, self.ms_size), interpolation=cv2.INTER_CUBIC)
            pan = cv2.resize(pan, (4 * self.ms_size, 4 * self.ms_size), interpolation=cv2.INTER_CUBIC)
            lpan = lpan[..., np.newaxis]
            pan = pan[..., np.newaxis]

        # random horizontal flip
        if self.hflip:
            lms = lms[:, ::-1, :]
            ms = ms[:, ::-1, :]
            lpan = lpan[:, ::-1, :]
            pan = pan[:, ::-1, :]

        # random vertical flip
        if self.vflip:
            lms = lms[::-1, :, :]
            ms = ms[::-1, :, :]
            lpan = lpan[::-1, :, :]
            pan = pan[::-1, :, :]

        # random rotate
        if self.rot:
            rot = random.randint(0, 3)
            lms = np.rot90(lms, rot, (0, 1))
            ms = np.rot90(ms, rot, (0, 1))
            lpan = np.rot90(lpan, rot, (0, 1))
            pan = np.rot90(pan, rot, (0, 1))

        return lms, ms, lpan, pan

    def np2tensor(self, x):
        # x shape: [H, W, C]
        # reshape to [C, H, W]
        ts = (2, 0, 1)
        x = torch.Tensor(x.transpose(ts).astype(float)).mul_(1.0)
        # normalization [-1,1]
        x = 2.0 * x / self.max_pixel - 1.0
        return x

    def __getitem__(self, index):
        lms = np.array(self.lms[index])
        ms = np.array(self.ms[index])
        lpan = np.array(self.lpan[index])
        pan = np.array(self.pan[index])

        if self.has_gt:
            gt = np.array(self.gt[index])
            if self.split == 'train':
                gt, lms, ms, lpan, pan = self.augment(gt, lms, ms, lpan, pan)
            return self.np2tensor(gt), self.np2tensor(lms), self.np2tensor(ms), self.np2tensor(lpan), self.np2tensor(pan)
        else:
            if self.split == 'train':
                lms, ms, lpan, pan = self.augment_without_gt(lms, ms, lpan, pan)
            return self.np2tensor(lms), self.np2tensor(ms), self.np2tensor(lpan), self.np2tensor(pan)

    def __len__(self):
        return self.pan.shape[0]