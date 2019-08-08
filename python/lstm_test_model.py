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
from matplotlib import pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')


# ! Manual Setting Const
kIsCompletelyTest = True
kIsErrorInspect = True

kBatchSize = 2048
kLoadModelNum = 11

kH5DataPath = os.path.join('..', 'data', 'h5data.same_mac')
kLogPath = os.path.join('.', 'log', 'tf.' + os.path.split(kH5DataPath)[1] + '.LSTM.log')

# ! Automatic Generated Const
kH5ModuleDataPath = os.path.join(kH5DataPath, 'h5_module_data')
kH5TrainTestDataPath = os.path.join(kH5DataPath, 'h5_train_test_split')
kSnapshotPath = os.path.join(kLogPath, 'snapshot', 'LSTM')
kTestResultPath = os.path.join(kLogPath, 'final_test_result')


def PlotWaveComparisonFig(gt_class_wave, predict_class_wave, error_waves_list):
    error_waves_num = len(error_waves_list)

    plt.figure()
    # ! Plot Reference Wave
    plt.subplot(2 + error_waves_num, 2, 1)
    plt.title('Ground Truth Class Wave')
    plt.plot(gt_class_wave)

    plt.subplot(2 + error_waves_list, 2, 2)
    plt.title('Predict Class Wave')
    plt.plot(predict_class_wave)
    # ! Plot Error Waves
    for i, error_wave in enumerate(error_waves_list):
        plt.subplot(2 + error_waves_list, 1, i + 2)
        plt.title('Error Wave {}'.format(i))
        plt.plot(error_wave)


def RandomSelectWaves(gt_class, predict_class, tester, batch_X, max_to_select=1):
    # ! Get waves' index list
    row = tester.classes_list.index(gt_class)
    col = tester.classes_list.index(predict_class)
    indexes = tester.confusion_list_matrix[row, col]
    # ! Random select indexes
    select_num = min(len(indexes), max_to_select)
    selected_indexes = np.random.choise(indexes, select_num, replace=False)
    return list(batch_X[selected_indexes, :])

def ErrorInspect(data_manager, sess, tester):
    # ! Get tensor X and Softmax probability
    input_X = graph.get_tensor_by_name('Placeholder:0')
    softmax_output = graph.get_tensor_by_name('Softmax:0')
    # ! Start Inspect
    while True:
        # ! Get a test batch then show test result
        tester.restart()
        batch_X, batch_Y = data_manager.get_random_test_samples(kBatchSize)
        batch_probability = sess.run(softmax_output, feed_dict={input_X: batch_X})
        tester.update_confusion_matrix(batch_probability, batch_Y)
        tester.show_confusion_matrix()
        # ! Decide whether use this test result
        print("Start Inspection input i; Retest input others")
        usr_select = input()
        if usr_select != 'i':
            continue
        else:
            # ! Start Inspect
            while True:
                # ! Read in gt and predict class name
                while True:
                    print("Input gt class name")
                    gt_class = input()
                    print("Input predict class name")
                    predict_class = input()
                    if gt_class not in data_manager.classes_list or predict_class not in data_manager.classes_list:
                        print("\033[1;31 Error: Input class name is not in class list, please input again")
                    else:
                        break
                if gt_class == predict_class:
                    print("\033[0;33 Warning: gt class is equal to predict class, no error wave will "
                          "be shown")
                # ! Start select waves and plot
                while True:
                    gt_class_wave = RandomSelectWaves(gt_class, gt_class,
                                                      tester, batch_X)[0]
                    predict_class_wave = RandomSelectWaves(predict_class, predict_class,
                                                           tester, batch_X)[0]
                    error_waves_list = RandomSelectWaves(gt_class, predict_class,
                                                         tester, batch_X, 2)
                    PlotWaveComparisonFig(gt_class_wave, predict_class_wave, error_waves_list)
                    # ! Interact Part
                    print("Quick replot input any key; Reselect classes input r; Retest input"
                          "t; Exit input e")
                    usr_select = input()
                    if usr_select != 'r' or usr_select != 't' or usr_select != 'e':
                        continue
                    else:
                        break
                if usr_select == 'r':
                    pass
                else:
                    break
            if usr_select == 't':
                pass
            else:
                break

def CompletelyTest(data_manager, graph, sess, tester):
    tester.restart()
    # ! Get tensor X and Softmax probability
    input_X = graph.get_tensor_by_name('Placeholder:0')
    softmax_output = graph.get_tensor_by_name('Softmax:0')
    # ! Start test data by batches
    test_batches = data_manager.get_test_batches(kBatchSize)
    # ! Start Test
    batch_num = int(np.ceil(data_manager.test_samples_num / kBatchSize))
    process_bar = ProcessBar(batch_num)
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


if __name__ == '__main__':
    # ! Init saver, sess, and data manager
    data_manager = DataManager(kH5TrainTestDataPath, kH5ModuleDataPath, I_only=True, down_sample=0)
    data_manager.init_epoch()

    saver = tf.train.import_meta_graph(kSnapshotPath + '-{}.meta'.format(kLoadModelNum))

    with tf.Session() as sess:
        # ! Restore graph, data, prepare tester
        saver.restore(sess, kSnapshotPath + '-{}'.format(kLoadModelNum))
        graph = tf.get_default_graph()
        tester = MultiClassificationTester(data_manager.classes_list)

        if kIsCompletelyTest:
            CompletelyTest(data_manager, graph, sess, tester)

        if kIsErrorInspect:
            ErrorInspect(data_manager, sess, tester)