#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>

import numpy as np
import torch as th
from torch import nn

class DQNModel(nn.Module):
    """ network model for deep Q-learning """
    def __init__(self, num_actions, shape_obser, num_obsers, method):
        super(DQNModel, self).__init__()
        self._num_actions = num_actions
        self._shape_obser = shape_obser
        self._num_obsers = num_obsers
        self._method = method
        
        self._hidden_len = np.prod(np.ceil(np.array(shape_obser) / 16)).astype("int")
        self._build_model()

    def _build_model(self):
        """ build network
        """
        self._base_conv = nn.Sequential(
            nn.Conv3d(in_channels=self._num_obsers*2, out_channels=32, kernel_size=4, stride=2, padding=1),# 2 channels input
            # nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            # nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=32, out_channels=46, kernel_size=2, stride=2, padding=0),
            # nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(num_features=46),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=46, out_channels=46, kernel_size=2, stride=2, padding=0)
            )

        if "Dueling" not in self._method:
            self._base_fc = nn.Sequential(
                nn.Linear(in_features=46*self._hidden_len, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.LeakyReLU(),
                nn.Linear(in_features=128, out_features=self._num_actions)
            )
        else:
            # Dueling DQN or Double Dueling
            # state value function
            self._state_fc = nn.Sequential(
                nn.Linear(in_features=46*self._hidden_len, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.LeakyReLU(),
                nn.Linear(in_features=128, out_features=1)
            )
            self._advan_fc = nn.Sequential(
                nn.Linear(in_features=46*self._hidden_len, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.LeakyReLU(),
                nn.Linear(in_features=128, out_features=self._num_actions)
            )

    def forward(self, x):
        x=th.Tensor.reshape(x,[x.shape[0],self._num_obsers*2,self._shape_obser[0],self._shape_obser[1],self._shape_obser[2]])
        base_conv = self._base_conv(x)
        base_conv = base_conv.view(x.size(0), -1)

        if "Dueling" not in self._method:
            Q = self._base_fc(base_conv)
        else:
            state_fc = self._state_fc(base_conv)
            advan_fc = self._advan_fc(base_conv)
            Q = state_fc + advan_fc - th.mean(advan_fc, dim=1, keepdim=True)
        
        return Q

    def initialization(self):
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
