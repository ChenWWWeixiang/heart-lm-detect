#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Utils.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>

import numpy as np
import torch as th
from torch.autograd import Variable
from collections import namedtuple

Statistic = namedtuple("Statistic",
                        ["reward_mean", "reward_std", "dist_mean", "dist_std", "step_mean", "step_std"])

def to_tensor_var(x, use_cuda=True, dtype=None):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    IntTensor = th.cuda.IntTensor if use_cuda else th.IntTensor
    CharTensor = th.cuda.CharTensor if use_cuda else th.CharTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor

    if dtype is None:
        return Variable(FloatTensor(x))
    elif dtype == "float32":
        return Variable(FloatTensor(x))
    elif dtype == "int64":
        return Variable(LongTensor(x))
    elif dtype == "int32":
        return Variable(IntTensor(x))
    elif dtype == "int8":
        return Variable(CharTensor(x))
    elif dtype == "uint8":
        return Variable(ByteTensor(x))

def unpacking_stati(rewards, infos):
    # reward
    reward_np = [np.mean(np.array(x), axis=0) for x in rewards]
    reward_mean = np.mean(reward_np, axis=0)
    reward_std = np.std(reward_np, axis=0)
    # distance error
    dist_np = [x[-1].dist_error for x in infos]
    dist_mean = np.mean(dist_np, axis=0)
    dist_std = np.std(dist_np, axis=0)
    # steps used
    step_np = [x[-1].steps for x in infos]
    step_mean = np.mean(step_np, axis=0)
    step_std = np.std(step_np, axis=0)

    return Statistic(reward_mean, reward_std, dist_mean, dist_std, step_mean, step_std)