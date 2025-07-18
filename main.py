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
import traceback
import argparse
import yaml

import torch
import numpy as np
import random

from train import Trainer
from utils import Report

class YamlAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        yaml_dict = yaml.safe_load(values)
        setattr(namespace, self.dest, yaml_dict)

def get_parser():
    parser = argparse.ArgumentParser(description='PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening')
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/test.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=2025, help='random seed for pytorch')
    parser.add_argument('--log-iter', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-iter', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-epoch', type=int, default=5, help='the interval for evaluating models (#iteration)')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--val-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for validation')
    parser.add_argument('--test-reduced-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for test (reduced)')
    parser.add_argument('--test-full-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for test (full)')

    # data
    parser.add_argument('--num-bands', type=int, default=4, help='the number of bands')
    parser.add_argument('--max-pixel', type=float, default=1.0, help='maximum pixel value')

    # model
    parser.add_argument('--model', default='model', help='the model will be used')
    parser.add_argument('--model-args', action=YamlAction, default=dict(), help='the arguments of model')

    # loss
    parser.add_argument('--res', type=bool, default=True, help='residual connection')
    parser.add_argument('--w-off', type=float, default=1.0, help='switcher threshold')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1'])

    # optim
    parser.add_argument('--gpu', type=int, default=0, help='the index of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--num-iter', type=int, default=1, help='the number of total iteration')
    parser.add_argument('--num-warmup', type=int, default=1, help='the number of warmup iteration')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
    parser.add_argument('--mixed-precision', type=str, default=None, choices=['no', 'fp16', 'bf16'])

    # test
    parser.add_argument('--pretrained-path', type=str, default='/model.safetensors', help='path for test')
    return parser


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(args):
    Feeder = import_class(args.feeder)
    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.train_feeder_args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True)
    data_loader['val'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.val_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    data_loader['test_reduced'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_reduced_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    data_loader['test_full'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_full_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    return data_loader


def load_model(args):
    Model = import_class(args.model)
    model = Model(**args.model_args)
    return model


def train(args):
    global_step = 0
    train_log = Report(args.work_dir, type='train')
    test_log = Report(args.work_dir, type='test')
    data_loader = load_data(args)
    model = load_model(args)

    trainer = Trainer(args=args, data_loader=data_loader, model=model)

    best_ergas = 9999
    best_ds = 1
    best_epoch_reduced = 0
    best_epoch_full = 0
    last_epoch = 0
    total_epoch = args.num_iter // len(data_loader['train']) + 1

    for epoch in range(last_epoch, total_epoch):
        train_log.write(f'========= Epoch {epoch + 1} of {total_epoch} =========')
        global_step = trainer.train(train_log, global_step)

        if (epoch + 1) % args.save_epoch == 0:
            trainer.save_checkpoint(epoch + 1)

        if (epoch + 1) % args.eval_epoch == 0:
            ergas = trainer.test_reduced(test_log, epoch + 1)
            ds = trainer.test_full(test_log, epoch + 1)
            if ergas < best_ergas:
                best_ergas = ergas
                best_epoch_reduced = epoch + 1
                trainer.save_best_model_reduced()
                trainer.test_reduced_save()
                trainer.test_full_save()
            if ds < best_ds:
                best_ds = ds
                best_epoch_full = epoch + 1
                trainer.save_best_model_full()
                trainer.test_reduced_save_full()
                trainer.test_full_save_full()

            test_log.write(f'Best ERGAS: {best_ergas:.6f}\tBest Epoch (Reduced): {best_epoch_reduced}\tBest D_s: {best_ds:.6f}\tBest Epoch (Full): {best_epoch_full}')

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    init_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.phase == 'train':
        train(args)
    else:
        raise ValueError('Unknown phase: {}'.format(args.phase))
