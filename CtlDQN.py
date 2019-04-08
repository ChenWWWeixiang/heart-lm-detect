import os
import random,time
import numpy as np
from collections import deque

from DataLoader import DataLoader
from MedEnv import MedEnv, ObserStack
from MedAgent import MedAgent
#from SimpleAgent import SimpleAgent as MedAgent
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
        self.update_config_per_epoch(0)
        while self._cnt_epoch < self._max_epoch:
            while self._cnt_frame < len(self._env._data_loader) and True:
                # interact
                #t1=time.time()
                self.interact()#2.6s
                #t2=time.time()
                #print(t2-t1)
                # training on a batch
                #if self._cnt_iter % (self._iters_per_update//100) == 0:
                self.update_batch()
                #t3 = time.time()
                #print(t3 - t2)
                # update target network
                #if self._cnt_iter % self._iters_per_update == 0:
                #    self.update_target_network()
                #print(self._env.location,self._env.angle)
            self.save_chkpoint()
            self.update_config_per_epoch(1)

            # evaluation
            if self._cnt_epoch % self._epochs_per_eval == 0:
                print("*"*50)
                print("[*] Perform evaluation")

                rewards, infos,locs = self.evaluate()
                stati = unpacking_stati(rewards, infos)
                
                print("Epoch: [{}] Average reward: {} Average score: {} Average step: {}".format(self._cnt_epoch - 1, stati.reward_mean, stati.dist_mean, stati.step_mean))

                self.save_chkpoint()
                print("*"*50)

    def play(self):
        tic=time.time()
        Di=[]
        self.load_chkpoint()
        re_go=1
        self.labeltxt='/home/data2/pan_cancer/all_results.txt'
        self.boxtxt='/home/data2/pan_cancer/labels.txt'
        f=open(self.labeltxt)
        f2=open(self.boxtxt)
        alldata2 = f2.readlines()
        name2 = [line.split(' ')[0] for line in alldata2]
        boxes = [line.split(' ')[0:] for line in alldata2]
        alldata=f.readlines()
        name=[line.split(' ')[0] for line in alldata]
        t = [line.split(' ')[1:4] for line in alldata]
        frames = deque(maxlen=self._num_obsers)
        f = open('results.txt', 'w')
        for i in range(len(self._env._data_loader)):
            Tt=[]
            for iter in range(re_go):
                if iter==0:
                    state = self._env.reset(False)
                else:
                    state = self._env.reset(True)
                for _ in range(self._num_obsers - 1):
                    frames.append(np.zeros_like(state))
                frames.append(state)
                print(self._env.moving.name.split('/')[-1]+' is coming!')
                matchname=self._env.moving.name.split('/')[-1].split('T2')[0]
                idx=name.index(matchname)
                idx2 = name2.index(matchname)
                B=boxes[idx2]
                T=t[idx]
                T=[int(i) for i in T]
                B = [int(i) for i in B[1:-2]]
                print('from '+str(self._env.offset)+' and initial Dis is '+str(self._env._calc_now_MI())+' target-->' +str(T))
                isOver = False
                cnt=0
                while not isOver:
                    action, qvalue = self._action(frames)
                    state, reward, isOver, info = self._env.step(action, qvalue)
                    frames.append(state)
                    print('step'+str(cnt)+' Q:'+str(qvalue)+' action:'+str(action)+' loc:'+str(self._env.offset)+'  Dis: '+str(self._env._calc_now_MI())+ '  reward: '+str(reward))
                    cnt+=1
                Tt.append(self._env.offset)
                print('name: '+self._env.moving.name+' loc:'+str(self._env.offset)+'  gtT:'+str(T)+'\n')
                f.writelines('name: '+self._env.moving.name+' loc:'+str(self._env.offset)+'  gtT:'+str(T)+'\n')
            T1box = B[:6]
            Tt=np.mean(Tt,0)
            Tt=np.concatenate([Tt,Tt])

            T2box = np.array(B[6:])+Tt*2
            dice=self.Dice3d(T2box,T1box)
            Di.append(dice)

        dice_av=np.mean(Di)
        acc=np.mean(np.array(Di)>0.5)
        print(dice_av,acc)
        print(time.time()-tic)




    def Dice3d(self,dets, gt_box):
        x = [dets[0], dets[3], gt_box[0], gt_box[3]]
        x.remove(np.max(x))
        x.remove(np.min(x))
        y = [dets[1], dets[4], gt_box[1], gt_box[4]]
        y.remove(np.max(y))
        y.remove(np.min(y))
        z = [dets[2], dets[5], gt_box[2], gt_box[5]]
        z.remove(np.max(z))
        z.remove(np.min(z))
        dice = 2 * (np.abs(x[0] - x[1]) * np.abs(y[0] - y[1]) * np.abs(z[0] - z[1])) / \
               (np.abs(dets[0] - dets[3]) * np.abs(dets[1] - dets[4]) * np.abs(dets[2] - dets[5]) +
                np.abs(gt_box[0] - gt_box[3]) * np.abs(gt_box[1] - gt_box[4]) * np.abs(gt_box[2] - gt_box[5]))
        if max(gt_box[0], gt_box[3]) < min(dets[0], dets[3]) or min(
                gt_box[0], gt_box[3]) > max(dets[0], dets[3]) or min(
            gt_box[1], gt_box[4]) > max(dets[1], dets[4]) or max(
            gt_box[1], gt_box[4]) < min(dets[1], dets[4]) or min(
            gt_box[2], gt_box[5]) > max(dets[2], dets[5]) or max(
            gt_box[2], gt_box[5]) < min(dets[2], dets[5]):
            dice = 0
        return dice
    def evaluate(self):
        rewards = []
        infos = []
        locs=[]

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
            locs.append(self._env_eval.env.offset)


        return rewards, infos,locs

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
                #self._args = chk_point["args"]
                self._cnt_epoch=chk_point["epoch"]
            except FileNotFoundError:
                print("Can\'t found checkpoint '{}'".format(self._model_path))
            print("[!] Load SUCCESS!")
        else:
            self._actor.initialization()
            print("[!] Load FAILD! Train from scratch!")
        # print training configuration
        print("[*] Training configuration list.")
        for key, value in self._args.items():
            print("{key}: {value}".format(key=key, value=value))
