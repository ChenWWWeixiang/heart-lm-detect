import os
import random
import numpy as np
from collections import deque

from DataLoader import DataLoader
from MedEnv import MedEnv, ObserStack
from MedAgent import MedAgent
from Utils import *

class CtlDQN(MedAgent):
    """ top class """
    def __init__(self,
                 phase,
                 batch_size,
                 shape_obser,
                 num_obsers,
                 num_actions,
                 iters_per_update,
                 reward_gamma,
                 max_memory_size,
                 init_memory_size,
                 drl_method,
                 max_epoch,
                 max_num_steps,
                 epochs_per_eval,
                 random_seed,
                 base_folder,
                 use_cuda,
                 model_path):
        super(CtlDQN, self).__init__(batch_size=batch_size,
                                     shape_obser=shape_obser,
                                     num_obsers=num_obsers,
                                     num_actions=num_actions,
                                     reward_gamma=reward_gamma,
                                     max_memory_size=max_memory_size,
                                     init_memory_size=init_memory_size,
                                     drl_method=drl_method,
                                     use_cuda=use_cuda)
        random.seed(random_seed)

        self._phase = phase
        self._iters_per_update = iters_per_update
        self._max_epoch = max_epoch
        self._max_num_steps = max_num_steps
        self._epochs_per_eval = epochs_per_eval
        self._base_folder = base_folder
        self._model_path = model_path

        self._args = {
            "batch_size": batch_size,
            "shape_obser": shape_obser,
            "num_obsers": num_obsers,
            "max_memory_size": max_memory_size,
            "iters_per_update": iters_per_update,
            "epochs_per_eval": epochs_per_eval,
            "reward_gamma": reward_gamma,
            "drl_method": drl_method,
            "max_num_steps": max_num_steps,
            "random_seed": random_seed
        }

        self.set_env(env=MedEnv(phase=self._phase,
                                base_folder=self._base_folder,
                                shape_obser=self._shape_obser,
                                num_actions=self._num_actions,
                                max_num_steps=self._max_num_steps))
        self._env_eval = ObserStack(env=MedEnv(phase="eval",
                                               base_folder=self._base_folder,
                                               shape_obser=self._shape_obser,
                                               num_actions=self._num_actions,
                                               max_num_steps=self._max_num_steps),
                                    num_obsers=self._num_obsers)

    def train(self):
        self.load_chkpoint()

        self.before_train(self._env)

        while self._cnt_epoch < self._max_epoch:
            while self._cnt_frame < len(self._env._data_loader):
                # interact
                self.interact()
                # training on a batch
                self.update_batch()
                # update target network
                if self._cnt_iter % self._iters_per_update == 0:
                    self.update_target_network()

            self.update_config_per_epoch()

            # evaluation
            if self._cnt_epoch % self._epochs_per_eval == 0:
                print("*"*50)
                print("[*] Perform evaluation")

                rewards, infos = self.evaluate()
                stati = unpacking_stati(rewards, infos)
                
                print("Epoch: [{}] Average reward: {} Average distance: {} Average step: {}".format(self._cnt_epoch - 1, stati.reward_mean, stati.dist_mean, stati.step_mean))

                self.save_chkpoint()
                print("*"*50)

    def play(self):
        pass

    def evaluate(self):
        rewards = []
        infos = []
        for i in range(len(self._env_eval.env._data_loader)):
            rewards_i = []
            infos_i = []
            state = self._env_eval.reset()
            isOver = False
            while not isOver:
                action, qvalue = self._action(state)
                state, reward, isOver, info = self._env_eval.step(action, qvalue)
                rewards_i.append(reward)
                infos_i.append(info)
            rewards.append(rewards_i)
            infos.append(infos_i)
        
        return rewards, infos

    def save_chkpoint(self):
        """ save model parameters into checkpoint model """
        print("[*] Saving checkpoint '{}' ...".format(self._model_path))
        th.save({
            "epoch": self._cnt_epoch,
            "args": self._args,
            "state_dict": self._actor.state_dict()
        }, self._model_path)
        print("[!] Saved!")

    def load_chkpoint(self):
        """ load model parameters from checkpoint model """
        print("[*] Loading checkpoint '{}' ...".format(self._model_path))
        if os.path.exists(self._model_path):
            try:
                chk_point = th.load(self._model_path)
                self._actor.load_state_dict(chk_point["state_dict"])
                self._args = chk_point["args"]
            except FileNotFoundError:
                print("Can\'t found checkpoint '{}'".format(self._model_path))
            print("[!] Load SUCCESS!")
        else:
            self._actor.initialization()
            print("[!] Load FAIELD! Train from scratch!")
        # print training configuration
        print("[*] Training configuration list.")
        for key, value in self._args.items():
            print("{key}: {value}".format(key=key, value=value))