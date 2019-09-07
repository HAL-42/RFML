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
from utils.data_manager import DataManager
from my_py_tools.my_logger import Logger, sh_logger
from my_py_tools.my_process_bar import ProcessBar
from utils.multi_classification_testor import MultiClassificationTester
from matplotlib import pyplot as plt
from data_process_patch import BatchCleaner

# ! Manual Setting Const
kIsCompletelyTest = True
kIsErrorInspect = True

kBatchSize = 2048
kLoadModelNum = 19

kH5DataPath = os.path.join('..', 'data', 'clean_h5data.diff_module_same_mac_mini5')
kLogPathComment = 'B32-lre-3'

kHotClean = False

kIOnly = True
# ! Automatic Generated Const
kH5ModuleDataPath = os.path.join(kH5DataPath, 'h5_module_data')
kH5TrainTestDataPath = os.path.join(kH5DataPath, 'h5_train_test_split')

kLogPath = os.path.join('.', 'log', 'tf.' + os.path.split(kH5DataPath)[1] + f'.LSTM.{kLogPathComment}.log')
kSnapshotPath = os.path.join(kLogPath, 'snapshot', 'LSTM')
kTestResultPath = os.path.join(kLogPath, f'final_test_result-{kLoadModelNum}')

def ZipIQ(batch_IQ):
    zip_batch_IQ = np.empty(batch_IQ.shape, dtype=np.float32)
    zip_batch_IQ[:, 0::2] = batch_IQ[:, :batch_IQ.shape[1] / 2]
    zip_batch_IQ[:, 1::2] = batch_IQ[:, batch_IQ.shape[1] / 2:]
    return zip_batch_IQ


def PlotWaveComparisonFig(gt_class_wave, predict_class_wave, error_waves_list):
    error_waves_num = len(error_waves_list)

    plt.figure()
    # ! Plot Ground Truth Reference Wave
    plt.subplot(2 + error_waves_num, 1, 1)
    plt.title('Ground Truth Class Wave')
    plt.plot(gt_class_wave)
    # ! Plot Error Waves
    for i, error_wave in enumerate(error_waves_list):
        plt.subplot(2 + error_waves_num, 1, i + 2)
        plt.title('Error Wave {}'.format(i))
        plt.plot(error_wave)
    # ! Plot Predict Reference Wave
    plt.subplot(2 + error_waves_num, 1, 2 + error_waves_num)
    plt.title('Predict Class Wave')
    plt.plot(predict_class_wave)

    plt.show()


def RandomSelectWaves(gt_class, predict_class, tester, data_manager, max_to_select=1):
    # ! Get waves' index list
    row = tester.classes_list.index(gt_class)
    col = tester.classes_list.index(predict_class)
    indexes = tester.confusion_list_matrix[row, col]
    # ! Random select indexes
    if len(indexes) == 0:
        return []
    else:
        select_num = min(len(indexes), max_to_select)
        selected_indexes = np.random.choice(indexes, select_num, replace=False)

        waves = [data_manager._get_sample('test', data_manager.shuffled_test_index[index])[0]
                 for index in selected_indexes]
        waves = np.array(waves, dtype=np.float32)
        if not kIOnly:
            waves = waves[:, :waves.shape[1] / 2] # Only return I data
        return waves


def ErrorInspect(data_manager, sess, tester):
    # ! Get tensor X and Softmax probability
    input_X = graph.get_tensor_by_name('Placeholder:0')
    softmax_output = graph.get_tensor_by_name('Softmax:0')
    # ! Start Inspect
    first_inspect = True
    while True:
        if first_inspect and kIsCompletelyTest:
            tester.show_confusion_matrix()
            first_inspect = False           # If not completely test, then is first inspect is unimportant
        else:
            # ! Get a test batch then show test result
            tester.restart()
            batch_X, batch_Y = data_manager.get_random_test_samples(kBatchSize)
            # if kHotClean:
            #     batch_X, batch_Y = BatchCleaner(batch_X, batch_Y)
            if not kIOnly:
                batch_X = ZipIQ(batch_X)
            batch_X = batch_X.reshape((batch_X.shape[0], input_X.shape[1], input_X.shape[2]))
            batch_probability = sess.run(softmax_output, feed_dict={input_X: batch_X})
            tester.update_confusion_matrix(batch_probability, batch_Y)
            tester.show_confusion_matrix()
        # ! Decide whether use this test result
        usr_select = input("Start Inspection input i; Retest input others: ")
        if usr_select != 'i':
            continue
        else:
            # ! Start Inspect
            while True:
                # ! Read in gt and predict class name
                while True:
                    gt_class = input("Input gt class name: ")
                    predict_class = input("Input predict class name: ")
                    if gt_class not in data_manager.classes_list or predict_class not in data_manager.classes_list:
                        print("\033[1;31mError: Input class name is not in class list, please input again \033[0m")
                    else:
                        break
                if gt_class == predict_class:
                    print("\033[0;33mWarning: gt class is equal to predict class, no error wave will "
                          "be shown \033[0m")
                # ! Start select waves and plot
                error_waves_select_num = 3
                while True:
                    gt_class_wave = RandomSelectWaves(gt_class, gt_class,
                                                      tester, data_manager)[0]
                    predict_class_wave = RandomSelectWaves(predict_class, predict_class,
                                                           tester, data_manager)[0]
                    error_waves_list = RandomSelectWaves(gt_class, predict_class,
                                                         tester, data_manager, error_waves_select_num)
                    PlotWaveComparisonFig(gt_class_wave, predict_class_wave, error_waves_list)
                    # ! Interact Part
                    usr_select = input("Quick replot input any key; Reselect classes input r; Retest input"
                          "t; Reset Error Waves Num input n; Exit input e: ")
                    if usr_select != 'r' and usr_select != 't' and usr_select != 'e' and usr_select != 'n':
                        pass
                    elif usr_select == 'n':
                        error_waves_select_num = int(input("Input Error Waves Num: "))
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
        # if kHotClean:
        #     batch_X, batch_Y = BatchCleaner(batch_X, batch_Y)
        if not kIOnly:
            batch_X = ZipIQ(batch_X)
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
    data_manager = DataManager(kH5TrainTestDataPath, kH5ModuleDataPath, I_only=kIOnly, down_sample=0)
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