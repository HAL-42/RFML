#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: inceptionresnet_1D_test_model.py
@time: 2019/9/2 17:42
@desc:
"""
import os
import numpy as np
import torch
from inceptionresnet_v2_model.data_manager import DataManager
from my_py_tools.my_logger import Logger, sh_logger
from my_py_tools.my_process_bar import ProcessBar
from inceptionresnet_v2_model.multi_classification_testor import MultiClassificationTester
from matplotlib import pyplot as plt
from data_process_patch import BatchCleaner
from inceptionresnet_v2_model.inceptionresnet_1D import InceptionResNet1D
from inceptionresnet_v2_model.torch_saver import Saver

# ! Manual Setting Const
kIsCompletelyTest = True
kIsErrorInspect = True

kBatchSize = 25
kLoadModelNum = -1
kTestSamplesNum = 1000

kH5DataDir = os.path.join('..', 'data', 'clean_h5data.diff_module_same_mac_mini5')
kLogDirComment = ''

kDevice = 'cpu'

# kHotClean = False

kIOnly = True
# ! Automatic Generated Const
kLogDir = os.path.join('.', 'log', f'torch.{os.path.split(kH5DataDir)[1]}.ICRS.{kLogDirComment}.log')
kH5ModuleDataDir = os.path.join(kH5DataDir, 'h5_module_data')
kH5TrainTestDataDir = os.path.join(kH5DataDir, 'h5_train_test_split')
kTestResultPath = os.path.join(kLogDir, 'final_test_result')
kSnapshotFileStr = os.path.join(kLogDir, 'snapshot', 'InceptionResNet1D-{}.snapshot')


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


def TestSamples(samples, gts, net, tester):
    sum_loss = 0
    i1 = 0
    while i1 < len(samples):
        if i1 + kBatchSize < len(samples):
            i2 = i1 + kBatchSize
        else:
            i2 = len(samples)
        batch_X = samples[i1:i2].reshape(i2 - i1, 1 if kIOnly else 2, -1)
        batch_X = torch.tensor(batch_X, dtype=torch.float32, device=kDevice)
        cpu_batch_Y = gts[i1:i2]
        batch_Y = torch.tensor(cpu_batch_Y, dtype=torch.float32, device=kDevice)
        with torch.no_grad():
            loss, PR = net.get_cross_entropy_loss(batch_X, batch_Y, need_PR=True, is_expanded_target=True)
        sum_loss += loss * (i2 - i1)
        tester.update_confusion_matrix(PR.cpu().numpy(), cpu_batch_Y)
        i1 += kBatchSize
    tester.measure()
    return float(sum_loss) / len(samples), tester.micro_avg_precision


def ErrorInspect(data_manager, net, tester):
    # ! Start Inspect
    first_inspect = True
    while True:
        if first_inspect and kIsCompletelyTest:
            tester.show_confusion_matrix()
        else:
            # ! Get a test batch then show test result
            tester.restart()
            samples, gts = data_manager.get_random_test_samples(kTestSamplesNum)
            # if kHotClean:
            #     batch_X, batch_Y = BatchCleaner(batch_X, batch_Y)
            TestSamples(samples, gts, net, tester)
        if first_inspect:
            first_inspect = False
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
            

def CompletelyTest(data_manager, net, tester):
    tester.restart()
    # ! Start test data by batches
    test_batches = data_manager.get_test_batches(kBatchSize)
    # ! Start Test
    batch_num = int(np.ceil(data_manager.test_samples_num / kBatchSize))
    process_bar = ProcessBar(batch_num)
    for i, test_batch in enumerate(test_batches):
        samples, gts = test_batch
        # if kHotClean:
        #     batch_X, batch_Y = BatchCleaner(batch_X, batch_Y)
        TestSamples(samples, gts, net, tester)
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
    data_manager = DataManager(kH5TrainTestDataDir, kH5ModuleDataDir, I_only=True, down_sample=0)
    data_manager.init_epoch()
    tester = MultiClassificationTester(data_manager.classes_list)
    saver = Saver(kSnapshotFileStr)

    net, _ = saver.restore(kLoadModelNum, model_cls=InceptionResNet1D, optimizer_cls=None, device=kDevice)
    net.to(kDevice)

    if kIsCompletelyTest:
        CompletelyTest(data_manager, net, tester)

    if kIsErrorInspect:
        ErrorInspect(data_manager, net, tester)