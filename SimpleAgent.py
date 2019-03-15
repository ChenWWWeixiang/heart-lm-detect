#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: MedAgent.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>

import numpy as np
import time
from tqdm import tqdm
from collections import namedtuple

from DQNModel import DQNModel
from ReplayMemory import ReplayMemory
from Utils import *

Experience = namedtuple("Experience",
                        ["state", "action", "reward", "isOver"])


class SimpleAgent(object):
    """ An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """

    def __init__(self,
                 batch_size,
                 shape_obser,
                 num_obsers,
                 num_actions,
                 reward_gamma,
                 max_memory_size,
                 init_memory_size,
                 drl_method,
                 optim_method="adam",
                 actor_lr=0.001,
                 use_cuda=True):
        super(SimpleAgent, self).__init__()
        self._batch_size = batch_size
        self._shape_obser = shape_obser
        self._num_obsers = num_obsers
        self._num_actions = num_actions
        self._reward_gamma = reward_gamma
        self._max_memory_size = max_memory_size
        self._init_memory_size = init_memory_size
        self._drl_method = drl_method
        self._optim_method = optim_method
        self._actor_lr = actor_lr
        self._use_cuda = use_cuda

        self._buff_iter = 0
        self._cnt_iter = 0
        self._cnt_frame = 0
        self._cnt_epoch = 0
        self._epsilon = 1.0
        self._frame_done = False
        self._memory = ReplayMemory(self._max_memory_size, self._shape_obser, self._num_obsers)

        self._actor = DQNModel(self._num_actions, self._shape_obser, self._num_obsers, self._drl_method)

        if self._use_cuda:
            self._actor.cuda()

    def set_env(self, env):
        self._env = env
        self._env_state = self._env.reset()

    def before_train(self, env):
        """ several manual initialization before training """
        self._init_memory()

        if self._optim_method == "adam":
            self._actor_optim = th.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        elif self._optim_method == "rmsprop":
            self._actor_optim = th.optim.RMSprop(self._actor.parameters(), lr=self._actor_lr)
        elif self._optim_method == "sgd":
            self._actor_optim = th.optim.SGD(self._actor.parameters(), lr=self._actor_lr)

        #self._target = DQNModel(self._num_actions, self._shape_obser, self._num_obsers, self._drl_method)
        #if self._use_cuda:
        #    self._target.cuda()
        #self.update_target_network()


    def update_batch(self):  # 7.5s
        """ train on a batch """
        start_time = time.time()

        # fetch a batch from memory
        batch = self._memory.sample(self._batch_size)
        state_var = to_tensor_var([x[:-1] for x in batch.state], self._use_cuda, dtype="float32")
        #next_state_var = to_tensor_var([x[1:] for x in batch.state], self._use_cuda, dtype="float32")
        action_var = to_tensor_var(batch.action, self._use_cuda, dtype="int64")
        reward_var = to_tensor_var(batch.reward, self._use_cuda, dtype="float32")
        #isOver_var = to_tensor_var(batch.isOver, self._use_cuda, dtype="float32")

        qvalue = self._actor.forward(state_var).gather(1, action_var[:, None])
        #next_qvalue = self._target.forward(next_state_var).detach()
        # compute target Q-value, using basic or Double algorithm
        target_qvalue = reward_var[:, None]

        # for debug

        # update
        self._actor_optim.zero_grad()
        loss = th.nn.SmoothL1Loss()(qvalue, target_qvalue)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self._actor.parameters(), max_norm=10)
        self._actor_optim.step()

        print(
            "Epoch: [{:<4d}] Iter: [{:<4d}] Env: [{:d}-{:<3d}] Speed: {:.2f}/sec Loss: {:.4f} Epsilon: {:.2f} Loc:{}".format(
                self._cnt_epoch, self._cnt_iter - self._buff_iter, self._cnt_frame, self._env.get_cnt(),
                self._batch_size / (time.time() - start_time), loss.item(), self._epsilon, self._env.location))

        # counter
        self._cnt_iter += 1
        if self._frame_done:
            self._cnt_frame += 1
            self._frame_done = False

    def update_config_per_epoch(self):
        # update counters
        self._buff_iter = self._cnt_iter + 1
        self._cnt_frame = 0
        self._cnt_epoch += 1
        # update epsilon
        turn_epoch_0 = 8
        turn_value_0 = 0.1
        turn_epoch_1 = 320
        turn_value_1 = 0.01
        if self._cnt_epoch <= turn_epoch_0:
            self._epsilon = (turn_value_0 - 1.0) * self._cnt_epoch / turn_epoch_0 + 1
        elif self._cnt_epoch <= turn_epoch_1:
            self._epsilon = (turn_value_1 - turn_value_0) * (self._cnt_epoch - turn_epoch_0) / (
                        turn_epoch_1 - turn_epoch_0) + turn_value_0

    def interact(self):
        return self._take_n_steps(10)

    def _take_one_step(self):
        return self._populate_exp()

    def _take_n_steps(self, n):
        for i in range(n):
            self._take_one_step()
        return

    def _init_memory(self):
        """ quickly fill the memory """
        print("[*] Initializing the experience replay set!")
        with tqdm(total=self._init_memory_size) as pbar:
            while len(self._memory) < self._init_memory_size:
                self._populate_exp()
                pbar.update()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy """
        curr_state = self._env_state

        qvalue = [0, ] * self._num_actions

        if np.random.random() <= self._epsilon:
            action = np.random.choice(self._num_actions)
        else:
            last_state = self._memory.recent_state()
            last_state.append(curr_state)

            action, qvalue = self._action(last_state)

        self._env_state, reward, isOver, _ = self._env.step(action, qvalue)

        if isOver:
            self._env_state = self._env.reset()
            self._frame_done = True

        self._memory.append(Experience(curr_state, action, reward, isOver))

    def _action(self, state):
        state_var = to_tensor_var(state, use_cuda=self._use_cuda)
        qvalue_var = self._actor.forward(state_var[None,])
        if self._use_cuda:
            qvalue = qvalue_var.data.cpu().numpy()[0]
        else:
            qvalue = qvalue_var.data.numpy()[0]
        action = np.argmax(qvalue)
        return action, qvalue
