#!/usr/bin python
# -*- coding: utf-8 -*-
# File: main.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>


import os
import sys
import time
import argparse
import numpy as np
import torch as th
from collections import deque

from CtlDQN import CtlDQN as DQN

###############################################################################
# batch size used in nature paper is 32 - medical is 256
BATCH_SIZE = 32
# breakout (84,84) - medical 2D (60,60) - medical 3D (26,26,26)
IMAGE_SIZE = (64,64,32)#(46, 46, 46)
# how many frames to keep
# in other words, how many observations the network can see
NUM_OBSERS = 4
# action space
NUM_ACTIONS = 6# 12 using 9 for debug
# discount factor - nature (0.99) - medical (0.9)
REWARD_GAMMA = 0.9
# replay memory size - nature (1e6) - medical (1e5 view-patches) debug 1e4
MAX_MEMORY_SIZE = 1e5
# initialization of memory buffer
INIT_MEMORY_SIZE = MAX_MEMORY_SIZE // 200  # 500
# max epochs
MAX_EPOCH = 1000
# the frequency of updating the target network
ITERS_PER_UPDATE = 2500 # 2.5k
# maximum number of steps per frame
MAX_NUM_STEPS = 1000
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 1
# random seed
RANDOM_SEED = 2019

# the directory containing data
BASE_PREFIX = "/mnt/data2/pan_cancer/"
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use", default="0")
    parser.add_argument("--method", help="comma separated list of type of DQN to use", default="Double, Duling")
    parser.add_argument("--phase", help="task to perform", choices=["train", "eval", "play"], default="train")
    parser.add_argument("--logdir", help="store logs in this directory during training", default="log")
    parser.add_argument("--name", help="name of current experiment for logs", default="DQN")

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        USE_CUDA = True
    else:
        USE_CUDA = False
    
    args.logdir = os.path.join("./train_log", args.logdir)
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    MODEL_PATH = os.path.join(args.logdir, args.name + ".pt")

    if args.phase != "play":
        BASE_FOLDER = os.path.join(BASE_PREFIX, "train")
    else:
        BASE_FOLDER = os.path.join(BASE_PREFIX, "test")

    model = DQN(phase=args.phase,
                batch_size=BATCH_SIZE,
                shape_obser=IMAGE_SIZE,
                num_obsers=NUM_OBSERS,
                num_actions=NUM_ACTIONS,
                iters_per_update=ITERS_PER_UPDATE,
                reward_gamma=REWARD_GAMMA,
                max_memory_size=MAX_MEMORY_SIZE,
                init_memory_size=INIT_MEMORY_SIZE,
                drl_method=args.method,
                max_epoch=MAX_EPOCH,
                max_num_steps=MAX_NUM_STEPS,
                epochs_per_eval=EPOCHS_PER_EVAL,
                random_seed=RANDOM_SEED,
                base_folder=BASE_FOLDER,
                use_cuda=USE_CUDA,
                model_path=MODEL_PATH)
    
    if args.phase == "train":
        model.train()## all in one
    elif args.phase == "play":
        model.play()
