#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DataReader.py
# Author: Yongjie Duan <dyj17@mails.tsinghua.edu.cn>

# import warnings
# warnings.simplefilter("ignore", category=ResourceWarning)

import os
import random
from glob import glob
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


class ImageRecord(object):
    """ a class to record image """
    pass


class CtlRecord(object):
    """ a class to record centerline """
    pass


class DataLoader(object):
    """ a class for managing image data """

    def __init__(self, phase="play", base_folder=None):
        super(DataLoader, self).__init__()
        # phase
        self.gtbox_path='/home/data2/pan_cancer/all_results.txt'
        f=open(self.gtbox_path)
        all=f.readlines()

        self.name=[a.split(' ')[0] for a in all]
        if True:
            gt = [a.split(' ')[1:] for a in all]
            self.GG=[]
            for igt in gt:
                x = int(igt[0])
                y = int(igt[1])
                z = int(igt[2])
                self.GG.append([x,y,z])
        self._phase = phase
        # folder containing images
        self._base_folder = base_folder

        # self._return_ctl = self._phase != "play"

        # read image list
        self.T2_list = glob("{}/*T2.mha".format(self._base_folder))
        self.T2_list.sort()
        # read ground truth if training or evaluation

        self.T1_list = glob("{}/*T1.mha".format(self._base_folder))
        self.T1_list.sort()

        # split into training and evaluation
        random.seed(2019)
        self._indexs = np.arange(len(self.T1_list))
        random.shuffle(self._indexs)
        if self._phase == "train":
            self._indexs = self._indexs[:-3]
        elif self._phase == "eval":
            self._indexs = self._indexs[-3:]


    def __len__(self):
        return len(self._indexs)

    @property
    def num_files(self):
        return len(self._indexs)

    def sample_circular(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        Keyword Arguments:
            shuffle {bool} -- [description] (default: {True})
        """
        while True:
            if shuffle:
                random.shuffle(self._indexs)
            for idx in self._indexs:
                fixed, moving = self._load_image_pair(idx)
                yield fixed, moving

    def _load_image_pair(self, idx):  #
        """ load image and centerline if training or evaluation
        Arguments:
            idx {[type]} -- [description]
        """
        image_sitk = sitk.ReadImage(self.T1_list[idx], sitk.sitkFloat32)
        image_np_T1 = sitk.GetArrayFromImage(image_sitk)
        image_np_T1 = resize(image_np_T1,
                             (image_np_T1.shape[0] // 2, image_np_T1.shape[1] // 2, image_np_T1.shape[2] // 2), order=3,
                             mode='constant', cval=0, clip=True, preserve_range=True)
        image_np = image_np_T1.swapaxes(0, 2)
        image_np = (image_np - image_np.mean()) / image_np.std()
        image_np = np.clip(image_np, -10, 10)

        fixed = ImageRecord()
        fixed.name = self.T1_list[idx]
        fixed.data = image_np
        fixed.shape = np.array(image_np.shape)
        fixed.origin = np.array(image_sitk.GetOrigin())
        fixed.spacing = np.array(image_sitk.GetSpacing())


        image_sitk = sitk.ReadImage(self.T2_list[idx], sitk.sitkFloat32)
        image_np_T2 = sitk.GetArrayFromImage(image_sitk)
        image_np_T2 = resize(image_np_T2,
                             (image_np_T2.shape[0] // 2, image_np_T2.shape[1] // 2, image_np_T2.shape[2] // 2), order=3,
                             mode='constant', cval=0, clip=True, preserve_range=True)
        zz = np.where(np.sum(np.sum(image_np_T2, 1), 1) > 0)
        image_np_T2 = image_np_T2[np.min(zz):np.max(zz), :, :]  # find the real volume area
        image_np = image_np_T2.swapaxes(0, 2)
        image_np = (image_np - image_np.mean()) / image_np.std()
        image_np = np.clip(image_np, -10, 10)

        moving = ImageRecord()
        moving.name = self.T2_list[idx]
        moving.data = image_np
        moving.shape = np.array(image_np.shape)
        moving.origin = np.array(image_sitk.GetOrigin())
        moving.spacing = np.array(image_sitk.GetSpacing())

        xid=self.name.index(self.T2_list[idx].split('/')[-1].split('T')[0])
        GT=self.GG[xid]
        moving.inital=GT
        return fixed, moving

    def _load_txt(self, txt_name):
        """ read txt file line by line and return vector containing all data
        Arguments:
            txt_name {[type]} -- [description]
        """
        data = np.loadtxt(os.path.join(txt_name, "vessel0", "reference.txt"), dtype="float", delimiter=' ')

        return data[:, :3]
