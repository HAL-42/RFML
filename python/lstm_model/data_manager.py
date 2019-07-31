#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: data_manager.py
@time: 2019/7/30 14:45
@desc:
"""
import os
import numpy as np
import json
from my_py_tools.my_exception import IllegalPhaseError
import h5py


class DataManager(object):
    def __init__(self, train_test_data_path, module_data_path, I_only = True, down_sample = 0):
        self.train_test_data_path = train_test_data_path
        self.module_data_path = module_data_path
        self.I_only = I_only
        self.down_sample = down_sample
        # ! Learn about train info
        with open(os.path.join(train_test_data_path, 'train_info.json'), 'r') as jf:
            train_info = json.load(jf)
            self.train_samples_num = train_info['sample_num']
            self.train_save_step = train_info['save_step']
            self.train_h5_names = train_info['h5_names']
        # ! Learn about test info
        with open(os.path.join(train_test_data_path, 'test_info.json'), 'r') as jf:
            test_info = json.load(jf)
            self.test_samples_num = test_info['sample_num']
            self.test_save_step = test_info['save_step']
            self.test_h5_names = test_info['h5_names']
        # ! Get class list
        self.classes_list = []
        for module_data_name in os.listdir(self.module_data_path):
            self.classes_list.append(module_data_name.split('.')[0])
        self.classes_num = len(self.classes_list)
        # ! Init train/test index
        self.shuffled_train_index = np.arange(0, self.train_samples_num)
        self.shuffled_test_index = np.arange(0, self.test_samples_num)

    def _label2y(self, label):
        y = np.zeros((self.classes_num,), dtype=np.int)
        y[self.classes_list.index(label)] = 1
        return y

    def _get_sample(self, phase:str, index):
        if phase ==  'train':
            # ! Find where the sample is
            h5_name_sample_in = self.train_h5_names[index % self.train_save_step]
            h5_path_sample_in = os.path.join(self.train_test_data_path, h5_name_sample_in)
            index_in_h5 = index - self.train_save_step * (index % self.train_save_step)
            # ! Read in sample's data
            with h5py.File(h5_path_sample_in, 'r') as hf:
                I_data = hf['I'][index_in_h5,:]
                Q_data = hf['Q'][index_in_h5,:]
                label = hf['Label'][index_in_h5]
            # label ---> Vector y
            y = self._label2y(label)
            # ! Return Sample Data
            if self.I_only:
                x = I_data
                if self.down_sample:
                    x = x[0::self.down_sample]
                return x, y
            else:
                if self.down_sample:
                    x = np.concatenate((I_data[0::self.down_sample], Q_data[0::self.down_sample]))
                else:
                    x = np.concatenate((I_data, Q_data))
                return x, y
        elif phase == 'test':
            # ! Find where the sample is
            h5_name_sample_in = self.test_h5_names[index % self.test_save_step]
            h5_path_sample_in = os.path.join(self.train_test_data_path, h5_name_sample_in)
            index_in_h5 = index - self.test_save_step * (index % self.test_save_step)
            # ! Read in sample's data
            with h5py.File(h5_path_sample_in, 'r') as hf:
                I_data = hf['I'][index_in_h5, :]
                Q_data = hf['Q'][index_in_h5, :]
                label = hf['Label'][index_in_h5]
            # label ---> Vector y
            y = self._label2y(label)
            # ! Return Sample Data
            if self.I_only:
                x = I_data
                if self.down_sample:
                    x = x[0::self.down_sample]
                return x, y
            else:
                if self.down_sample:
                    x = np.concatenate((I_data[0::self.down_sample], Q_data[0::self.down_sample]))
                else:
                    x = np.concatenate((I_data, Q_data))
                return x, y
        else:
            raise IllegalPhaseError("Phase Parameter Should be train or test")

    def init_epoch(self):
        np.random.shuffle(self.shuffled_train_index)
        np.random.shuffle(self.shuffled_test_index)

    def get_train_batches(self, batch_size):
        # ! Init Generator
        i1 = 0
        # ! Start Generator
        while i1 < self.train_samples_num:
            if i1 + batch_size < self.train_samples_num:
                i2 = i1 + batch_size
            else:
                i2 = self.train_samples_num
            # ! Yield batch from i1 ---- i2
            xs = []
            ys = []
            for index in self.shuffled_train_index[i1:i2]:
                x, y = self._get_sample('train', index)
                xs.append(x)
                ys.append(y)
            yield (np.array(xs), np.array(ys))

            i1 = i2
        return

    def get_test_batches(self, batch_size):
        # ! Init Generator
        i1 = 0
        # ! Start Generator
        while i1 < self.test_samples_num:
            if i1 + batch_size < self.test_samples_num:
                i2 = i1 + batch_size
            else:
                i2 = self.test_samples_num
            # ! Yield batch from i1 ---- i2
            xs = []
            ys = []
            for index in self.shuffled_test_index[i1:i2]:
                x, y = self._get_sample('test', index)
                xs.append(x)
                ys.append(y)
            yield (np.array(xs), np.array(ys))

            i1 = i2
        return

    def get_random_test_samples(self, samples_num):
        assert samples_num <= self.test_samples_num, "Sample Num > Test Samples Num"
        np.random.shuffle(self.shuffled_test_index)
        xs = []
        ys = []
        for index in self.shuffled_test_index[:samples_num]:
            x, y = self._get_sample('test', index)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)