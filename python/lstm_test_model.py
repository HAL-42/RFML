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
from lstm_model.data_manager import DataManager
from my_py_tools.my_logger import Logger, sh_logger
from my_py_tools.my_process_bar import ProcessBar
from lstm_model.multi_classification_testor import MultiClassificationTester

kBatchSize = 1024
kLoadModelNum = 11

kH5DataPath = os.path.join('..', 'data', 'h5data.same_mac')
kH5ModuleDataPath = os.path.join(kH5DataPath, 'h5_module_data')
kH5TrainTestDataPath = os.path.join(kH5DataPath, 'h5_train_test_split')
kLogPath = os.path.join('.', 'log', 'tf.' + os.path.split(kH5DataPath)[1] + '.LSTM.log')
kSnapshotPath = os.path.join(kLogPath, 'snapshot', 'LSTM')
kTestResultPath = os.path.join(kLogPath, 'final_test_result')

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
        tester = MultiClassificationTester(data_manager.classes_list)
        for i, test_batch in enumerate(test_batches):
            batch_X, batch_Y = test_batch
            batch_X = batch_X.reshape((batch_X.shape[0], input_X.shape[1], input_X.shape[2]))
            batch_probability = sess.run(softmax_output, feed_dict={input_X: batch_X})

            tester.update_confusion_matrix(batch_probability, batch_Y)
            tester.show_confusion_matrix()

            process_bar.UpdateBar(i + 1)
        # ! Show test result
        if not os.path.isdir(kTestResultPath):
            os.makedirs(kTestResultPath)
        tester.show_confusion_matrix(img_save_path=os.path.join(kTestResultPath, "confusion_matrix.png"))
        tester.measure()
        tester.show_measure_result(rslt_save_path=os.path.join(kTestResultPath, "test_result.txt"))