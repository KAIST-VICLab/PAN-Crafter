# --------------------------------------------------------
# PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening
# Copyright (c) 2025 Jeonghyeok Do, Sungpyo Kim, Geunhyuk Youk, Jaehyup Lee†, and Munchurl Kim†
#
# This code is released under the MIT License (see LICENSE file for details).
#
# This software is licensed for **non-commercial research and educational use only**.
# For commercial use, please contact: mkimee@kaist.ac.kr
# --------------------------------------------------------

import os
import sys
import torch
import numpy as np
from einops import rearrange
from pathlib import Path

from numpy.linalg import norm
import cv2
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage.filters import sobel, convolve
from scipy.stats import pearsonr

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpatialCorrelationCoefficient
import lpips
from pytorch_fid import fid_score
from torchmetrics.image.inception import InceptionScore


def to_rgb(x, tol_low=0.01, tol_high=0.99):
    x = (x + 1.0) / 2.0
    x = torch.Tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        has_batch = False
        x = x.unsqueeze(0)
    else:
        has_batch = True
    # Try to detect BCHW or BHWC
    if x.shape[1] > 8:
        x = rearrange(x, 'b h w c -> b c h w')
    c = x.shape[1]
    if c == 1:
        x = torch.cat([x, x, x], dim=1)
    elif c == 3:
        pass
    elif c == 4:
        x = x[:, [2, 1, 0], :, :]
    elif c == 8:
        x = x[:, [4, 2, 1], :, :]
    else:
        raise ValueError(f"Unsupported channel number: {c}")
    b, c, h, w = x.shape
    x = rearrange(x, 'b c h w -> c (b h w)')
    sorted_x, _ = torch.sort(x, dim=1)
    t_low = sorted_x[:, int(b * h * w * tol_low)].unsqueeze(1)
    t_high = sorted_x[:, int(b * h * w * tol_high)].unsqueeze(1)
    x = torch.clamp((x - t_low) / (t_high - t_low), 0, 1)
    x = rearrange(x, 'c (b h w) -> b h w c', b=b, c=c, h=h, w=w)
    if not has_batch:
        x = x.squeeze(0)
    return x.cpu().numpy()


def tensor2img(tensor, max_pixel):
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.squeeze(0).float().cpu().clamp(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # H W C
    img_np = (img_np * max_pixel).round()
    return img_np / max_pixel


def SSIM_numpy(x_true, x_pred, data_range=1.0):
    return structural_similarity(x_true, x_pred, data_range=data_range, channel_axis=-1)


def MPSNR_numpy(x_true, x_pred, data_range=1.0):
    tmp = []
    for c in range(x_true.shape[-1]):
        tmp.append(peak_signal_noise_ratio(x_true[:, :, c], x_pred[:, :, c], data_range=data_range))
    return np.mean(tmp)


def SAM_numpy(x_true, x_pred):
    dot_sum = np.sum(x_true * x_pred, axis=2)
    norm_true = norm(x_true, axis=2)
    norm_pred = norm(x_pred, axis=2)
    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    sam = np.mean(res)
    return sam * 180 / np.pi


def SCC_numpy(x_true, x_pred):
    # Sobel edge detection for each channel
    Im_Lap_F = np.sqrt(sobel(x_pred[1:-1, 1:-1, :], axis=0) ** 2 + sobel(x_pred[1:-1, 1:-1, :], axis=1) ** 2)
    Im_Lap_GT = np.sqrt(sobel(x_true[1:-1, 1:-1, :], axis=0) ** 2 + sobel(x_true[1:-1, 1:-1, :], axis=1) ** 2)

    # Compute SCC (Spectral Correlation Coefficient)
    numerator = np.sum(Im_Lap_F * Im_Lap_GT)
    denominator = np.sqrt(np.sum(Im_Lap_F ** 2) * np.sum(Im_Lap_GT ** 2))
    return np.mean(numerator / (denominator + 1e-8))  # Avoid division by zero


def SCC_full_numpy(x_true, x_pred):
    x_true_expanded = np.repeat(x_true, x_pred.shape[2], axis=2)

    # Sobel edge detection for each channel
    Im_Lap_F = np.sqrt(sobel(x_pred[1:-1, 1:-1, :], axis=0) ** 2 + sobel(x_pred[1:-1, 1:-1, :], axis=1) ** 2)
    Im_Lap_GT = np.sqrt(sobel(x_true_expanded[1:-1, 1:-1, :], axis=0) ** 2 + sobel(x_true_expanded[1:-1, 1:-1, :], axis=1) ** 2)

    # Compute SCC for each channel
    numerator = np.sum(Im_Lap_F * Im_Lap_GT, axis=(0, 1))  # Sum over spatial dimensions
    denominator = np.sqrt(np.sum(Im_Lap_F ** 2, axis=(0, 1)) * np.sum(Im_Lap_GT ** 2, axis=(0, 1)))
    return np.mean(numerator / (denominator + 1e-8))


def Q4_numpy(x_true, x_pred):
    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0, :] = 1
        return a * sign

    def product(a, b):
        a = a.reshape(a.shape[0], 1)
        b = b.reshape(b.shape[0], 1)
        R = np.dot(a, b.transpose())
        r = np.zeros(4)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
        r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
        r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
        return r

    if x_true.shape[2] > 4:
        x_true, x_pred = x_true[:, :, :4], x_pred[:, :, :4]

    imps = np.copy(x_pred)
    imms = np.copy(x_true)
    vec_ps = imps.reshape(imps.shape[1] * imps.shape[0], imps.shape[2])
    vec_ps = vec_ps.transpose(1, 0)
    vec_ms = imms.reshape(imms.shape[1] * imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1, 0)
    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1, 0) - m1).transpose(1, 0)
    s1 = np.mean(np.sum(d1 * d1, axis=0))
    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)
    s2 = np.mean(np.sum(d2 * d2, axis=0))
    Sc = np.zeros(vec_ms.shape)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:, i] = product(d1[:, i], d2[:, i])
    C = np.mean(Sc, axis=1)
    Q4 = 4 * np.sqrt(np.sum(m1 * m1) * np.sum(m2 * m2) * np.sum(C * C)) / (s1 + s2) / (
            np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q4


def RMSE_numpy(x_true, x_pred):
    d = (x_true - x_pred) ** 2
    rmse = np.sqrt(np.sum(d) / (d.shape[0] * d.shape[1]))
    return rmse


def ERGAS_numpy(x_true, x_pred, ratio=0.25):
    c = x_true.shape[2]
    summed = 0.0
    for i in range(c):
        summed += (RMSE_numpy(x_true[:, :, i], x_pred[:, :, i])) ** 2 / np.mean(x_pred[:, :, i]) ** 2
    ergas = 100 * ratio * np.sqrt(summed / c)
    return ergas


def ERGAS_full_numpy(x_true, x_pred, ratio=0.25):
    h, w, c = x_true.shape
    x_pred_down = cv2.resize(x_pred, (w, h), interpolation=cv2.INTER_CUBIC)

    summed = 0.0
    for i in range(c):
        summed += (RMSE_numpy(x_true[:, :, i], x_pred_down[:, :, i])) ** 2 / np.mean(x_pred_down[:, :, i]) ** 2
    ergas = 100 * ratio * np.sqrt(summed / c)
    return ergas


def QIndex_numpy(a, b):
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    temp = np.cov(a, b)
    d1 = temp[0, 0]
    cov = temp[0, 1]
    d2 = temp[1, 1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4 * cov * m1 * m2 / (d1 + d2) / (m1 ** 2 + m2 ** 2)
    return Q


def UIQC_numpy(x_true, x_pred):
    c = x_true.shape[2]
    uiqc = 0.0
    for i in range(c):
        uiqc += QIndex_numpy(x_true[:, :, i], x_pred[:, :, i])
    return uiqc / c


def D_lambda_numpy(ms, ps):
    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += np.abs(QIndex_numpy(ps[:, :, i], ps[:, :, j]) - QIndex_numpy(ms[:, :, i], ms[:, :, j]))
    return sum / L / (L - 1)


def D_s_numpy(ms, pan, ps):
    H, W, L = ms.shape
    l_pan = cv2.resize(pan, (W, H), interpolation=cv2.INTER_CUBIC)
    sum = 0.0
    for i in range(L):
        sum += np.abs(QIndex_numpy(ps[:, :, i], pan) - QIndex_numpy(ms[:, :, i], l_pan))
    return sum / L


def QNR_numpy(ms, pan, ps):
    d_lambda = D_lambda_numpy(ms, ps)
    d_s = D_s_numpy(ms, pan, ps)
    qnr = (1 - d_lambda) * (1 - d_s)
    return qnr


def write(log, str):
    sys.stdout.flush()
    log.write(str + '\n')
    log.flush()


def reduced_metrics(x_true, x_pred, max_pixel):
    x_true, x_pred = tensor2img(x_true, max_pixel), tensor2img(x_pred, max_pixel)
    metrics = dict()
    metrics['psnr'] = MPSNR_numpy(x_true, x_pred)
    metrics['ssim'] = SSIM_numpy(x_true, x_pred)
    metrics['scc'] = SCC_numpy(x_true, x_pred)
    metrics['sam'] = SAM_numpy(x_true, x_pred)
    metrics['q4'] = Q4_numpy(x_true, x_pred)
    metrics['ergas'] = ERGAS_numpy(x_true, x_pred)
    return metrics


def full_metrics(x_pred, pan, ms, max_pixel):
    x_pred, pan, ms = tensor2img(x_pred, max_pixel), tensor2img(pan, max_pixel), tensor2img(ms, max_pixel)
    metrics = dict()
    metrics['d_lambda'] = D_lambda_numpy(ms, x_pred)
    metrics['d_s'] = D_s_numpy(ms, pan, x_pred)
    metrics['qnr'] = QNR_numpy(ms, pan, x_pred)
    metrics['ergas'] = ERGAS_full_numpy(pan, x_pred)
    metrics['scc'] = SCC_full_numpy(pan, x_pred)
    return metrics


class Report():
    def __init__(self, save_dir, type):
        filename = os.path.join(save_dir, f'{type}_log.txt')

        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(filename):
            self.logFile = open(filename, 'a')
        else:
            self.logFile = open(filename, 'w')

    def write(self, str):
        print(str)
        write(self.logFile, str)

    def __del__(self):
        self.logFile.close()


class Train_Report():
    def __init__(self):
        self.total_loss = []
        self.loss_ms = []
        self.loss_pan = []
        self.num_examples = 0

    def update(self, batch_size, total_loss, loss_ms, loss_pan):
        self.num_examples += batch_size
        self.total_loss.append(total_loss * batch_size)
        self.loss_ms.append(loss_ms * batch_size)
        self.loss_pan.append(loss_pan * batch_size)

    def compute_mean(self):
        self.total_loss = np.sum(self.total_loss) / self.num_examples
        self.loss_ms = np.sum(self.loss_ms) / self.num_examples
        self.loss_pan = np.sum(self.loss_pan) / self.num_examples

    def result_str(self, lr, period_time):
        self.compute_mean()
        str = f'Total Loss: {self.total_loss:.7f}\tLoss MS: {self.loss_ms:.7f}\tLoss PAN: {self.loss_pan:.7f}\tLearning Rate: {lr:.7f}\tTime: {period_time:.4f}'
        return str


class Test_Reduced_Report():
    def __init__(self):
        self.psnr = []
        self.ssim = []
        self.scc = []
        self.sam = []
        self.q4 = []
        self.ergas = []
        self.num_examples = 0

    def update(self, batch_size, metrics):
        self.num_examples += batch_size
        self.psnr.append(metrics['psnr'])
        self.ssim.append(metrics['ssim'])
        self.scc.append(metrics['scc'])
        self.sam.append(metrics['sam'])
        self.q4.append(metrics['q4'])
        self.ergas.append(metrics['ergas'])

    def compute_mean(self):
        self.psnr = np.sum(self.psnr) / self.num_examples
        self.ssim = np.sum(self.ssim) / self.num_examples
        self.scc = np.sum(self.scc) / self.num_examples
        self.sam = np.sum(self.sam) / self.num_examples
        self.q4 = np.sum(self.q4) / self.num_examples
        self.ergas = np.sum(self.ergas) / self.num_examples

    def result_str(self):
        self.compute_mean()
        str = f'PSNR: {self.psnr:.6f}\tSSIM: {self.ssim:.6f}\tSCC: {self.scc:.6f}\t'
        str += f'SAM: {self.sam:.6f}\tQ4: {self.q4:.6f}\tERGAS: {self.ergas:.6f}'
        return str


class Test_Full_Report():
    def __init__(self):
        self.d_lambda = []
        self.d_s = []
        self.qnr = []
        self.ergas = []
        self.scc = []
        self.num_examples = 0

    def update(self, batch_size, metrics):
        self.num_examples += batch_size
        self.d_lambda.append(metrics['d_lambda'])
        self.d_s.append(metrics['d_s'])
        self.qnr.append(metrics['qnr'])
        self.ergas.append(metrics['ergas'])
        self.scc.append(metrics['scc'])

    def compute_mean(self):
        self.d_lambda = np.sum(self.d_lambda) / self.num_examples
        self.d_s = np.sum(self.d_s) / self.num_examples
        self.qnr = np.sum(self.qnr) / self.num_examples
        self.ergas = np.sum(self.ergas) / self.num_examples
        self.scc = np.sum(self.scc) / self.num_examples

    def result_str(self):
        self.compute_mean()
        str = f'D_lambda: {self.d_lambda:.6f}\tD_s: {self.d_s:.6f}\tQNR: {self.qnr:.6f}\t'
        str += f'ERGAS: {self.ergas:.6f}\tSCC: {self.scc:.6f}'
        return str