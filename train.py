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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from functools import partial
from safetensors.torch import load_file

from tqdm import tqdm
import numpy as np
from scipy.io import savemat

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler

from utils import to_rgb, reduced_metrics, full_metrics, Train_Report, Test_Reduced_Report, Test_Full_Report


class Trainer:
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']
        self.test_full_data_loader = data_loader['test_full']

        # Accelerator
        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            project_config=self.accelerator_project_config)

        if self.accelerator.is_main_process:
            if args.work_dir is not None:
                os.makedirs(args.work_dir, exist_ok=True)

        # Weight data type
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Model
        self.model = model

        # Optimizer
        params_to_opt = self.model.parameters()
        self.optimizer = torch.optim.AdamW(params_to_opt, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        # Scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup,
            num_training_steps=self.args.num_iter)

        # Accelerator
        self.model = self.accelerator.prepare(self.model)

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.args.work_dir, f"epoch-{epoch}")
        self.accelerator.save_state(save_path)

    def save_best_model_reduced(self):
        save_path = os.path.join(self.args.work_dir, 'best_reduced')
        self.accelerator.save_state(save_path)

    def save_best_model_full(self):
        save_path = os.path.join(self.args.work_dir, 'best_full')
        self.accelerator.save_state(save_path)

    def save_val(self, pan, gt, gen, lms, idx):
        path = os.path.join(self.args.work_dir, 'results/val/')
        if not os.path.exists(path):
            os.makedirs(path)

        pan, gt, gen, lms = to_rgb(pan), to_rgb(gt), to_rgb(gen), to_rgb(lms)
        generated_1 = np.concatenate((pan, lms), axis=2)
        generated_2 = np.concatenate((gen, gt), axis=2)
        generated_image = np.concatenate((generated_1, generated_2), axis=1)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(f'{path}/{idx:04}.png')

    def save_test_reduced(self, pan, gt, gen, lms, idx):
        path = os.path.join(self.args.work_dir, 'results/reduced/')
        if not os.path.exists(path):
            os.makedirs(path)

        pan, gt, gen, lms = to_rgb(pan), to_rgb(gt), to_rgb(gen), to_rgb(lms)
        generated_1 = np.concatenate((pan, lms), axis=2)
        generated_2 = np.concatenate((gen, gt), axis=2)
        generated_image = np.concatenate((generated_1, generated_2), axis=1)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(f'{path}/{idx:04}.png')

    def save_test_full(self, pan, gen, lms, idx):
        path = os.path.join(self.args.work_dir, 'results/full/')
        if not os.path.exists(path):
            os.makedirs(path)

        pan, gen, lms = to_rgb(pan), to_rgb(gen), to_rgb(lms)
        generated_image = np.concatenate((pan, lms, gen), axis=2)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(f'{path}/{idx:04}.png')

    def train(self, train_log, global_step):
        self.model.train()
        self.model.requires_grad_(True)
        report = Train_Report()
        start = time.time()

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    gt = gt.to(self.accelerator.device, dtype=self.weight_dtype).repeat(2, 1, 1, 1)
                    lms = lms.to(self.accelerator.device, dtype=self.weight_dtype).repeat(2, 1, 1, 1)
                    ms = ms.to(self.accelerator.device, dtype=self.weight_dtype).repeat(2, 1, 1, 1)
                    lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype).repeat(2, 1, 1, 1)
                    pan = pan.to(self.accelerator.device, dtype=self.weight_dtype).repeat(2, 1, 1, 1)

                    res_pan = F.interpolate(lpan, scale_factor=4, mode="bicubic")
                    res_ms = F.interpolate(ms, scale_factor=4, mode="bicubic")

                    switch_off = torch.zeros((self.args.batch_size,), device=self.accelerator.device).to(dtype=self.weight_dtype)
                    switch_on = torch.ones((self.args.batch_size,), device=self.accelerator.device).to(dtype=self.weight_dtype)
                    switch = torch.cat((switch_off, switch_on), dim=0)

                objective_recon = self.model(pan, lpan, ms, switch)

                if self.args.res:
                    objective_recon = objective_recon + res_ms * switch.view(-1, 1, 1, 1) + res_pan.repeat(1, self.args.num_bands, 1, 1) * (1.0 - switch).view(-1, 1, 1, 1)

                if self.args.loss_type == 'l1':
                    loss_pan = (pan[:self.args.batch_size].repeat(1, self.args.num_bands, 1, 1) - objective_recon[:self.args.batch_size]).abs().mean() * self.args.w_off
                    loss_ms = (gt[self.args.batch_size:] - objective_recon[self.args.batch_size:]).abs().mean()
                    loss = loss_pan + loss_ms
                else:
                    raise NotImplementedError()

                reduced_loss = self.accelerator.gather(loss).mean()
                reduced_loss_ms = self.accelerator.gather(loss_ms).mean()
                reduced_loss_pan = self.accelerator.gather(loss_pan).mean()

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size * 2, reduced_loss.item(), reduced_loss_ms.item(), reduced_loss_pan.item())

            global_step += 1

            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                result_str = report.result_str(lr, period_time)
                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

            if global_step % self.args.save_iter == 0:
                save_path = os.path.join(self.args.work_dir, f'checkpoint-{global_step}')
                self.accelerator.save_state(save_path)

            if global_step >= self.args.num_iter:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                result_str = report.result_str(lr, period_time)
                train_log.write(prefix_str + result_str)
                save_path = os.path.join(self.args.work_dir, 'lastest')
                self.accelerator.save_state(save_path)
                self.accelerator.end_training()
                return global_step

        return global_step

    def test_reduced(self, test_log, epoch):
        report = Test_Reduced_Report()
        self.model.eval()
        self.model.requires_grad_(False)

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    generated = self.model(pan, lpan, ms, switch) + F.interpolate(ms, scale_factor=4, mode="bicubic")
                else:
                    generated = self.model(pan, lpan, ms, switch)
                self.save_test_reduced(pan, gt, generated, F.interpolate(ms, scale_factor=4, mode="bicubic"), idx)
                metrics = reduced_metrics(x_true=gt, x_pred=generated, max_pixel=self.args.max_pixel)
                report.update(self.args.test_batch_size, metrics)

        prefix_str = f'Epoch[{epoch}]\t'
        result_str = report.result_str()
        test_log.write(prefix_str + result_str)
        return report.ergas

    def test_full(self, test_log, epoch):
        report = Test_Full_Report()
        self.model.eval()
        self.model.requires_grad_(False)
        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    generated = self.model(pan, lpan, ms, switch) + F.interpolate(ms, scale_factor=4, mode="bicubic")
                else:
                    generated = self.model(pan, lpan, ms, switch)
                self.save_test_full(pan, generated, F.interpolate(ms, scale_factor=4, mode="bicubic"), idx)
                metrics = full_metrics(x_pred=generated, pan=pan, ms=ms, max_pixel=self.args.max_pixel)
                report.update(self.args.test_batch_size, metrics)

        prefix_str = f'Epoch[{epoch}]\t'
        result_str = report.result_str()
        test_log.write(prefix_str + result_str)
        return report.d_s

    def test_reduced_save(self):
        self.model.eval()
        self.model.requires_grad_(False)

        gt_list = []
        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    generated = self.model(pan, lpan, ms, switch) + F.interpolate(ms, scale_factor=4, mode="bicubic")
                else:
                    generated = self.model(pan, lpan, ms, switch)
                gt_list.append(gt)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        gt_save = (torch.cat(gt_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=gt_save * self.args.max_pixel,
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/reduced.mat', d)

    def test_full_save(self):
        self.model.eval()
        self.model.requires_grad_(False)

        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    generated = self.model(pan, lpan, ms, switch) + F.interpolate(ms, scale_factor=4, mode="bicubic")
                else:
                    generated = self.model(pan, lpan, ms, switch)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/full.mat', d)

    def test_reduced_save_full(self):
        self.model.eval()
        self.model.requires_grad_(False)

        gt_list = []
        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    generated = self.model(pan, lpan, ms, switch) + F.interpolate(ms, scale_factor=4, mode="bicubic")
                else:
                    generated = self.model(pan, lpan, ms, switch)
                gt_list.append(gt)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        gt_save = (torch.cat(gt_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=gt_save * self.args.max_pixel,
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/reduced_full.mat', d)

    def test_full_save_full(self):
        self.model.eval()
        self.model.requires_grad_(False)

        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    generated = self.model(pan, lpan, ms, switch) + F.interpolate(ms, scale_factor=4, mode="bicubic")
                else:
                    generated = self.model(pan, lpan, ms, switch)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/full_full.mat', d)