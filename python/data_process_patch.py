#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: data_process_patch.py
@time: 2019/8/16 6:14
@desc:
"""
import numpy as np
import h5py
from my_py_tools.my_logger import sh_logger, Logger
import os
import glob
from matplotlib import pyplot as plt
from data_processing import GetH5TrainTestData
from my_py_tools.my_process_bar import ProcessBar

# ! Set Manually
kTailInHeadThreshold = 0.8
kSampleInBatchMeanThreshold = 0.35
kHeadTailPercentage = 0.4

kIsDebug = False
kMaxToKeepWaves = 8

# kH5DataDir = os.path.join('..', 'data', 'h5data.diff_module_same_mac_mini5')
kH5DataDir = "F:\\h5data.diff_module_same_mac_43"
kLogFile = os.path.join('.', 'log')

kIsChar2Num = True
kChar2NumDict = {'A':'36', 'B': '37', 'C':'38', 'D':'39', 'E':'40', 'F':'41', 'G':'42', 'H':'43'}


def PlotSamples(samples):
    # ! Decide show samples
    if samples.shape[0] == 0:
        print("\033[0;33mWarning: No Waves to Plot\033[0m")
        return
    elif samples.shape[0] > kMaxToKeepWaves:
        print("\033[0;33mWarning: Randomly Select {} of {} Waves to Plot\033[0m"
              .format(kMaxToKeepWaves, samples.shape[0]))
        selected_indexes = np.random.choice(samples.shape[0], kMaxToKeepWaves, replace=False)
        show_samples = samples[selected_indexes]
    else:
        show_samples = samples
    # ! Show Samples
    plt.figure()
    for i in range(show_samples.shape[0]):
        plt.subplot(show_samples.shape[0], 1, i + 1)
        plt.plot(show_samples[i, :])
    plt.show()
    input("-------Press Any Key to Continue--------")

def BatchCleaner(samples, labels):
    samples_energy = samples ** 2
    head_tail_len = int(samples.shape[1] * kHeadTailPercentage)
    # ! Delete Samples whose tail energy is less than head's 80%
    del_indexes = []
    for i, sample_energy in enumerate(samples_energy):
        if np.mean(sample_energy[-head_tail_len:]) < kTailInHeadThreshold * np.mean(sample_energy[0:head_tail_len]):
            del_indexes.append(i)
    if kIsDebug:
        print("Plot Samples Deleted for tail < 0.8 * head")
        PlotSamples(samples[del_indexes, :])
    samples = np.delete(samples, del_indexes, axis=0)
    labels = np.delete(labels, del_indexes, axis=0)
    samples_energy = np.delete(samples_energy, del_indexes, axis=0)
    # ! Delete Samples whose mean energy is too small
    samples_mean_energy = np.mean(samples_energy)
    del_indexes = []
    for i, sample_energy in enumerate(samples_energy):
        if np.mean(sample_energy) < kSampleInBatchMeanThreshold * samples_mean_energy:
            del_indexes.append(i)
    if kIsDebug:
        print("Plot Samples Deleted for sample_mean < 0.8 * samples_mean")
        PlotSamples(samples[del_indexes, :])
    samples = np.delete(samples, del_indexes, axis=0)
    labels = np.delete(labels, del_indexes, axis=0)
    return samples, labels


def CleanModuleData(h5_module_data_dir, clean_h5_module_data_dir):
    logger = Logger(os.path.join(kLogFile, 'clean_module_data.txt')).logger
    clean_h5_module_data_txt_file = os.path.join(
        clean_h5_module_data_dir, 'h5_module_data.txt')
    # ! Process each wifi module
    module_data_num = dict()
    for wifi_module_file in glob.glob(os.path.join(h5_module_data_dir, '*.h5')):
        module_name = os.path.split(wifi_module_file)[1].split('.')[0]
        logger.info("Currently Processing Wifi Module {}".format(module_name))
        if kIsChar2Num:
            if module_name in kChar2NumDict:
                module_name = kChar2NumDict[module_name]
                logger.info("Replace Wifi Module Name to {}".format(module_name))
        # ! Skip if Already Porcessed
        if os.path.isfile(os.path.join(clean_h5_module_data_dir, module_name + '.h5')):
            logger.warning("{}.h5 Already Exits, Skip".format(module_name))
            continue
        # ! Get Raw Data
        with h5py.File(wifi_module_file) as hf:
            I_data = hf['I'][...]
            Q_data = hf['Q'][...]
        # ! Clean Data
        I_energies = I_data ** 2
        Q_energies = Q_data ** 2
        head_tail_len = int(I_data.shape[1] * kHeadTailPercentage)
        # ! Delete Samples whose tail energy is less than head's 80%
        del_indexes = []
        sh_logger.debug("Start Quite Tail Discarding")
        process_bar = ProcessBar(I_energies.shape[0] - 1)
        for i in range(I_energies.shape[0]):
            I_energy = I_energies[i, :]
            Q_energy = Q_energies[i, :]
            if np.mean(I_energy[-head_tail_len:]) < kTailInHeadThreshold * np.mean(I_energy[0:head_tail_len])\
                    or \
                np.mean(Q_energy[-head_tail_len:]) < kTailInHeadThreshold * np.mean(Q_energy[0:head_tail_len]):
                del_indexes.append(i)
            process_bar.UpdateBar(i)
        if kIsDebug:
            print("Plot Samples Deleted for tail < 0.8 * head")
            PlotSamples(I_data[del_indexes, :])
            PlotSamples(Q_data[del_indexes, :])
        I_data = np.delete(I_data, del_indexes, axis=0)
        Q_data = np.delete(Q_data, del_indexes, axis=0)
        I_energies = np.delete(I_energies, del_indexes, axis=0)
        Q_energies = np.delete(Q_energies, del_indexes, axis=0)
        # ! Delete Samples whose mean energy is too small
        I_mean_energy = np.mean(I_energies)
        Q_mean_energy = np.mean(Q_energies)
        del_indexes = []
        sh_logger.debug("Start Whole Quite Discarding")
        process_bar = ProcessBar(I_energies.shape[0] - 1)
        for i in range(I_energies.shape[0]):
            I_energy = I_energies[i, :]
            Q_energy = Q_energies[i, :]
            if np.mean(I_energy) < kSampleInBatchMeanThreshold * I_mean_energy\
                    or \
               np.mean(Q_energy) < kSampleInBatchMeanThreshold * Q_mean_energy   :
                del_indexes.append(i)
            process_bar.UpdateBar(i)
        if kIsDebug:
            print("Plot Samples Deleted for sample_mean < 0.8 * samples_mean")
            PlotSamples(I_data[del_indexes, :])
            PlotSamples(Q_data[del_indexes, :])
        I_data = np.delete(I_data, del_indexes, axis=0)
        Q_data = np.delete(Q_data, del_indexes, axis=0)
        labels = np.array([module_name] * I_data.shape[0], dtype=np.object)
        # ! Save to h5 file
        sh_logger.debug("Start Saving H5")
        with h5py.File(os.path.join(clean_h5_module_data_dir, module_name + '.h5')) as hf:
            hf.create_dataset('I', data=I_data)
            hf.create_dataset('Q', data=Q_data)
            dset_labels = hf.create_dataset('Label', labels.shape, dtype=h5py.special_dtype(vlen=str))
            dset_labels[:] = labels
        # ! Record in txt
        module_data_num[module_name] = I_data.shape[0]
        with open(clean_h5_module_data_txt_file, 'a') as txt_f:
            txt_f.write("{} {}\n".format(
                module_name, module_data_num[module_name]))
    return module_data_num


if __name__ == '__main__':
    # ! Prepare path
    clean_h5_data_dir_name = 'clean_' + os.path.split(kH5DataDir)[1]
    clean_h5_data_dir = os.path.join(os.path.split(kH5DataDir)[0], clean_h5_data_dir_name)
    if not os.path.isdir(clean_h5_data_dir):
        os.mkdir(clean_h5_data_dir)

    h5_module_data_dir = os.path.join(kH5DataDir, 'h5_module_data')
    # h5_train_test_data_dir = os.path.join(kH5DataDir, 'h5_train_test_split')
    clean_h5_module_data_dir = os.path.join(clean_h5_data_dir, 'h5_module_data')
    clean_h5_train_test_data_dir = os.path.join(clean_h5_data_dir, 'h5_train_test_split')

    module_data_num = None
    if not os.path.isdir(clean_h5_module_data_dir):
        os.mkdir(clean_h5_module_data_dir)
        module_data_num = CleanModuleData(h5_module_data_dir, clean_h5_module_data_dir)

    if not os.path.isdir(clean_h5_train_test_data_dir):
        os.mkdir(clean_h5_train_test_data_dir)
        GetH5TrainTestData(clean_h5_module_data_dir,
                           clean_h5_train_test_data_dir, module_data_num)
    else:
        sh_logger.warning("Dir {} is Already Exits. Skip Func GetTrainH5TestData".
                          format(clean_h5_train_test_data_dir))