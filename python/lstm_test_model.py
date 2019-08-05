#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: lstm_test_model.py
@time: 2019/7/31 17:05
@desc:
"""
import os
import numpy as np
import tensorflow as tf
from lstm_model.build_model import BuildModel
from lstm_model.data_manager import DataManager
from my_py_tools.my_logger import Logger, sh_logger
from my_py_tools.my_process_bar import ProcessBar
import time

kBatchSize = 1024
kLoadModelNum = 11

kH5DataPath = os.path.join('..', 'data', 'h5data.same_mac')
kH5ModuleDataPath = os.path.join(kH5DataPath, 'h5_module_data')
kH5TrainTestDataPath = os.path.join(kH5DataPath, 'h5_train_test_split')
kLogPath = os.path.join('.', 'log', 'tf.' + os.path.split(kH5DataPath)[1] + '.LSTM.log')
kSnapshotPath = os.path.join(kLogPath, 'snapshot', 'LSTM')

if __name__ == '__main__':
    # ! Init saver, sess, and data manager
    data_manager = DataManager(kH5TrainTestDataPath, kH5ModuleDataPath, I_only=True, down_sample=0)
    data_manager.init_epoch()

    saver = tf.train.import_meta_graph(kSnapshotPath + '-{}.meta'.format(kLoadModelNum))

    with tf.Session() as sess:
        # ! Restore graph and data
        saver.restore(sess, kSnapshotPath + '-{}'.format(kLoadModelNum))
        # ! Get tensor X and Softmax probability
        graph = tf.get_default_graph()
        input_X = graph.get_tensor_by_name('Placeholder:0')
        softmax_output = graph.get_tensor_by_name('Softmax:0')
        # ! Start test data by batches
        test_batches = data_manager.get_test_batches(kBatchSize)

        batch_num = int(np.ceil(data_manager.test_samples_num / kBatchSize))
        process_bar = ProcessBar(batch_num)
        for i, test_batch in enumerate(test_batches):
            batch_X, batch_Y = test_batch
            batch_probability = sess.run(softmax_output, feed_dict={input_X: batch_X})

            process_bar.SkipMsg(str(batch_probability[0:5, :]), sh_logger)
            process_bar.SkipMsg(str(batch_Y[0:5, :]), sh_logger)

            process_bar.UpdateBar(i + 1)