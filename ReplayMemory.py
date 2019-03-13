#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ExpReplay.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>

import numpy as np
import random
import copy
from collections import deque, namedtuple

Experience = namedtuple("Experience",
                        ["state", "action", "reward", "isOver"])

class ReplayMemory(object):
    """ replay memory buffer """
    def __init__(self, max_size, shape_obser, num_obsers):
        super(ReplayMemory, self).__init__()
        self._max_size = int(max_size)
        self._shape_obser = shape_obser
        self._num_obsers = int(num_obsers)

        self._state = np.zeros((self._max_size,2) + self._shape_obser, dtype="float32")
        self._action = np.zeros((self._max_size,), dtype="int8")
        self._reward = np.zeros((self._max_size,), dtype="float32")
        self._isOver = np.zeros((self._max_size,), dtype="bool")

        self._curr_size = 0
        self._curr_pos = 0
        self._obser_history = deque(maxlen=self._num_obsers - 1)

    def __len__(self):
        return self._curr_size
        
    def append(self, exp):
        """ append the replay memory with experience sample        
        Arguments:
            exp {[type]} -- [description]
        """
        # increase current memory size if it is not full yet
        self._assign(self._curr_pos, exp)
        
        if self._curr_size < self._max_size:
            self._curr_size += 1
        
        self._curr_pos = (self._curr_pos + 1) % self._max_size

        if exp.isOver:
            self._obser_history.clear()
        else:
            self._obser_history.append(exp.state)

    def sample(self, batch_size):
        batch_idx = random.sample(range(self._curr_size - self._num_obsers - 1), batch_size)
        batch_exp = [self._fetch_data(i) for i in batch_idx]
        return Experience(*zip(*batch_exp))

    def recent_state(self):
        """ return a list of recent state: (num_obsers-1,) + state_size """
        lst = list(self._obser_history)
        states = [np.zeros([2,self._shape_obser[0],self._shape_obser[1],self._shape_obser[2]], dtype="float32")] * (self._num_obsers - 1 - len(lst))
        states.extend([k for k in lst])
        return states

    # def _process_batch(self, batch_exp):
    #     state = np.asarray([e.state for e in batch_exp], dtype='float32')
    #     action = np.asarray([e.action for e in batch_exp], dtype='int8')
    #     reward = np.asarray([e.reward for e in batch_exp], dtype='float32')
    #     isOver = np.asarray([e.isOver for e in batch_exp], dtype='bool')
    #     return Experience(state, action, reward, isOver)

    def _fetch_data(self, idx):
        """ sample an experience replay from memory with given index     
        Arguments:
            idx {[type]} -- [description]
        """
        idx = (self._curr_pos + idx) % self._curr_size
        k = self._num_obsers + 1
        if idx + k <= self._curr_size:
            state = self._state[idx:idx + k]
            action = self._action[idx:idx + k]
            reward = self._reward[idx:idx + k]
            isOver = self._isOver[idx:idx + k]
        else:
            end = idx + k - self._curr_size
            state = self._slice(self._state, idx, end)
            action = self._slice(self._action, idx, end)
            reward = self._slice(self._reward, idx, end)
            isOver = self._slice(self._isOver, idx, end)

        return self._pad_sample(state, action, reward, isOver)

    def _assign(self, pos, exp):
        self._state[pos] = exp.state
        self._action[pos] = exp.action
        self._reward[pos] = exp.reward
        self._isOver[pos] = exp.isOver

    def _slice(self, arr, start, end):
        s1 = arr[start:self._curr_size]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def _pad_sample(self, state, action, reward, isOver):
        """ the next_state is a different episode if current_state.isOver==True        
        Arguments:
            state {[type]} -- [description]
            reward {[type]} -- [description]
            action {[type]} -- [description]
            isOver {bool} -- [description]
        Returns:
            [type] -- [description]
        """
        for k in range(self._num_obsers - 2, -1, -1):
            if isOver[k]:
                state = copy.deepcopy(state)
                state[:k + 1].fill(0)
                break

        return Experience(state, action[-2], reward[-2], isOver[-2])
