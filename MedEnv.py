#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: MedEnv.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>

import numpy as np
import gym
from skimage.transform import resize
from sklearn import metrics as mr
from gym import spaces
from collections import Counter, deque, namedtuple

from DataLoader import DataLoader

Info = namedtuple("Info",
                        ["dist_error", "steps"])

class MedEnv(gym.Env):
    """ a class represents enviroment
    Arguments:
        gym {[type]} -- [description]
    Returns:
        [type] -- [description]
    """
    def __init__(self,
                 phase,
                 base_folder,
                 shape_obser,
                 num_actions,
                 max_num_steps=1000,
                 step_history_len=10):
        super(MedEnv, self).__init__()
        # phase
        self._phase = phase
        # base folder containing image data
        self._base_folder = base_folder
        # the shape of observation
        self._shape_obser = shape_obser
        # maximum number of steps per episodes
        self._max_num_steps = max_num_steps
        # the number of actions
        self._num_actions = num_actions
        # counter to limit number of steps per image
        self._cnt = 0
        #initial angle
        self.angle=0
        # 3D or ...
        self._ndims = 2*len(self._shape_obser)
        # history buffer for storing last locations to check oscilations
        self._step_history_len = step_history_len
        self._reset_history()
        # actions, only translation
        self._action_trans = np.concatenate((np.eye(self._ndims), -1 * np.eye(self._ndims)), axis=0).astype("int")
        #(dx,dy,dz,dangle1,dangle2,dangle3)
        # data loader
        self._data_loader = DataLoader(phase=self._phase, base_folder=self._base_folder)
        self._data_sampler = self._data_loader.sample_circular(shuffle=self._phase!="play")

        # self.action_space = spaces.Discrete(self._num_actions)
        # self.observation_space = spaces.Box(low=-10, high=10, shape=self._shape_obser)

    def get_cnt(self):
        return self._cnt

    def step(self, action, qvalue):
        """ take an action on the current state
        Arguments:
            action {[type]} -- [description]
            qvalue {[type]} -- [description]
        Returns:
            [type] -- [description]
        """
        go_out = False
        self._qvalue = qvalue
        self._isOver = False

        curr_location = self.location
        curr_angle=self.angle

        [next_location,next_angle] = [curr_location,curr_angle] + self._action_trans[action]
        if np.any(next_location < np.array([0, 0, 0])) or np.any(next_location >= np.array(self.fixed.shape)):
            next_location = curr_location
            go_out = True

        # punish -1 reward if the agent tries to go out

        if go_out:
            reward = -1
        else:
            reward = np.clip(self._calc_reward(curr_location, next_location,curr_angle,next_angle), -1, 1)

        # update
        self.location = next_location
        self.angle=next_angle
        self.state = self._get_state_current()
        self._dist_current = self._calc_mutualInformation(self.location)

        # terminate if the agent reached the last point
        if self._phase == "train" and self._dist_current <= 0.5:##TODO: maybe the parameter 0.5 need to be changed
            self._isOver = True

        # terminate if maximum number of steps is reached
        self._cnt += 1
        if self._cnt >= self._max_num_steps:
            self._isOver = True

        # update history buffer with new location
        self._update_history()

        # check if agent oscillates
        if self._phase != "train" and self._oscillate:
            self._isOver = True
            self.location = self._get_location_best()
            self.angle=self._get_get_angle_best()
            self.state = self._get_state_current()

        # update distance between fixed and moving



        return self.state, reward, self._isOver, Info(self._dist_current, self._cnt)

    def reset(self):
        """ reset state and anything related
        Returns:
            [type] -- [description]
        """
        self._isOver = False
        self._cnt = 0
        self._reset_history()##

        # init a fixed and a moving volumn
        self.fixed, self.moving = next(self._data_sampler)
        # image volume size
        self._shape_image_fixed = self.fixed.shape
        self._shape_image_moving = self.moving.shape
        ##start from a radom location
        self.location=np.transpose(np.array([np.random.randint(self._shape_obser[i]//2,self._shape_image_fixed[i]-self._shape_obser[i]/2,1)
                                for i in range(3)]),[1,0])[0]
        self.angle=np.transpose(np.array([np.random.randint(-5,5,1)for i in range(3)]),[1,0])[0]##TODO:need to add rotate
        # self.location = np.array(self._ctl.start_point)
        #self.location = np.array([np.random.randint(x - 15, x + 15, dtype = "int") for x in self.moving.end_point])
        self.state = self._get_state_current()

        self._dist_current = self._calc_mutualInformation(self.location,self.angle)

        return self.state
    def _calc_mutualInformation(self,loc,ang):##TODO: ang has no been used
        half_size_l = np.array(self._shape_image_moving, dtype="int") // 2
        half_size_r = np.array(self._shape_image_moving, dtype="int") - half_size_l

        bbox_l_tmp = loc - half_size_l
        bbox_r_tmp = loc + half_size_r

        # check if they violate image boundary and fix them
        bbox_l = np.max((bbox_l_tmp, np.array([0, 0, 0])), axis=0)
        bbox_r = np.min((bbox_r_tmp, np.array(self._shape_image_fixed)), axis=0)
        area=bbox_r-bbox_l
        area=area[0]*area[1]*area[2]/(self._shape_image_moving[0]*self._shape_image_moving[1]*self._shape_image_moving[2])
        state_fixed=self.fixed.data[bbox_l[0]:bbox_r[0], bbox_l[1]:bbox_r[1], bbox_l[2]:bbox_r[2]]
        moving_l = np.max([bbox_l - loc+self._shape_image_moving//2, np.array([0, 0, 0])], axis=0)
        moving_r = np.max([bbox_r - loc+self._shape_image_moving//2, self._shape_image_moving//2- loc+self._shape_image_fixed], axis=0)
        state_moving = self.moving.data[moving_l[0]:moving_r[0], moving_l[1]:moving_r[1], moving_l[2]:moving_r[2]]
        MI=mr.normalized_mutual_info_score(np.reshape(state_fixed,[-1]),np.reshape(state_moving,[-1]))

        return MI+area
    def _calc_reward(self, curr_location, next_location,curr_ang,next_ang):
        """ calculate the reward based on the decrease in MI to the end point
        Arguments:
            curr_location {[type]} -- [description]
            next_location {[type]} -- [description]
        """
        MI_curr = self._calc_mutualInformation(curr_location,curr_ang)
        MI_next = self._calc_mutualInformation(next_location,next_ang)

        return MI_curr - MI_next
    def _get_state_current(self):##TODO: ang has no been used
        """ crop image data around current location to obtain what network sees
        """
        # initialization

        current_angle=self.angle##TODO: ang has no been used

        half_size_l = np.array(self._shape_image_moving, dtype="int") // 2
        half_size_r = np.array(self._shape_image_moving, dtype="int") - half_size_l

        bbox_l_tmp = self.location - half_size_l
        bbox_r_tmp = self.location + half_size_r

        # check if they violate image boundary and fix them
        bbox_l = np.max((bbox_l_tmp, np.array([0, 0, 0])), axis=0)
        bbox_r = np.min((bbox_r_tmp, np.array(self._shape_image_fixed)), axis=0)
        state_fixed=self.fixed.data[bbox_l[0]:bbox_r[0], bbox_l[1]:bbox_r[1], bbox_l[2]:bbox_r[2]]
        moving_l = np.max([bbox_l - self.location+self._shape_image_moving//2, np.array([0, 0, 0])], axis=0)
        moving_r = np.max([bbox_r - self.location+self._shape_image_moving//2, self._shape_image_moving//2- self.location+self._shape_image_fixed], axis=0)
        state_moving = self.moving.data[moving_l[0]:moving_r[0], moving_l[1]:moving_r[1], moving_l[2]:moving_r[2]]
        state_moving=resize(state_moving,(self._shape_obser[0],self._shape_obser[1],self._shape_obser[2]),order=3,
                            mode='constant',cval=0,clip=True,preserve_range=True)
        state_fixed=resize(state_fixed,(self._shape_obser[0],self._shape_obser[1],self._shape_obser[2]),order=3,
                            mode='constant',cval=0,clip=True,preserve_range=True)
        state=np.stack([state_fixed,state_moving],axis=0)#shape=(2,x,y,z)

        return state

    def _calc_distance(self, point1, point2, spacing=[1,1,1]):
        """ calculate the distance between two points in mm """
        return np.linalg.norm((point1 - point2)*spacing)

    def _update_history(self):
        """ update history buffer with current state
        """
        # update location history
        self._loc_history[:-1] = self._loc_history[1:]
        self._loc_history[-1] = self.location
        #update angle history
        self._ang_history[:-1] = self._ang_history[1:]
        self._ang_history[-1] = self.angle
        # update Q-value history
        self._qvalue_history[:-1] = self._qvalue_history[1:]
        self._qvalue_history[-1] = self._qvalue

    def _reset_history(self):
        self._loc_history = np.zeros([self._step_history_len, self._ndims], dtype="int")
        self._ang_history = np.zeros([self._step_history_len, self._ndims], dtype="int")
        self._qvalue_history = np.zeros([self._step_history_len, self._num_actions])

    @property
    def _oscillate(self):
        """ return True if the agent is stuck and oscillating
        """
        unique, counts = np.unique(self._loc_history, axis=0, return_counts=True)
        counts_sort = counts.argsort()
        if np.all(unique[counts_sort[-1]] == [0, 0, 0]):
            if counts[counts_sort[-2]] > 3:
                return True
            else:
                return False
        elif counts[counts_sort[-1]] > 3:
            return True

    def _get_location_best(self):
        """ get the best location with the best Q value from locations stored in history
        """
        last_qvalue_history = self._qvalue_history[-4:]
        last_loc_history = self._loc_history[-4:]
        best_qvalue = np.max(last_qvalue_history, axis=1)
        best_idx = best_qvalue.argmin()
        best_location = last_loc_history[best_idx]

        return best_location

    def _get_angle_best(self):
        """ get the best location with the best Q value from locations stored in history
        """
        last_qvalue_history = self._qvalue_history[-4:]
        last_ang_history = self._ang_history[-4:]
        best_qvalue = np.max(last_qvalue_history, axis=1)
        best_idx = best_qvalue.argmin()
        best_angle = last_ang_history[best_idx]

        return best_angle
# =============================================================================
# ================================ ObserStack =================================
# =============================================================================

class ObserStack(gym.Wrapper):##TODO:What is it?
    """ used when not training, wrapper for MedEnv """
    def __init__(self, env, num_obsers):
        super(ObserStack, self).__init__(env)
        self._num_obsers = num_obsers
        self.frames = deque(maxlen=self._num_obsers)

    def reset(self):
        obser = self.env.reset()
        for _ in range(self._num_obsers - 1):
            self.frames.append(np.zeros_like(obser))
        self.frames.append(obser)
        return self.frames

    def step(self, action, qvalue):
        obser, reward, isOver, info = self.env.step(action, qvalue)
        self.frames.append(obser)
        return self.frames, reward, isOver, info
