#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: environment_noise_analysis.py
@time: 2019/9/8 12:17
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
import json

kH5DataDir = "F:\\h5data.diff_module_same_mac_43"
kH5ModuleDataDir = os.path.join(kH5DataDir, 'h5_module_data')
kLogFile = os.path.join('.', 'log')
# ! Set Manually
kTailInHeadThreshold = 0.8
kSampleInBatchMeanThreshold = 0.35
kHeadTailPercentage = 0.4

kMaxToKeepWaves = 8

kHeadLen = 1000
kTailLen = 1000

kIsDebug = False


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
    plt.figure(dpi=300)
    for i in range(show_samples.shape[0]):
        plt.subplot(show_samples.shape[0], 1, i + 1)
        plt.plot(show_samples[i, :])
    plt.show()
    input("-------Press Any Key to Continue--------")

def GetSNRs(h5_module_data_dir):
    # ! Process each wifi module
    module_data_num = dict()
    named_SNRs = {}
    for wifi_module_file in glob.glob(os.path.join(h5_module_data_dir, '*.h5')):
        module_name = os.path.split(wifi_module_file)[1].split('.')[0]
        sh_logger.info("Currently Processing Wifi Module {}".format(module_name))

        # ! Get Raw Data
        with h5py.File(wifi_module_file) as hf:
            I_data = hf['I'][...]
            Q_data = hf['Q'][...]
        # ! Get SNR
        I_energies = I_data ** 2
        Q_energies = Q_data ** 2
        head_tail_len = int(I_data.shape[1] * kHeadTailPercentage)
        # Get unbalance samples
        indexes_unbalance = []
        sh_logger.debug("Detect Unbalanced")
        process_bar = ProcessBar(I_energies.shape[0] - 1)
        for i in range(I_energies.shape[0]):
            I_energy = I_energies[i, :]
            Q_energy = Q_energies[i, :]
            if np.mean(I_energy[-head_tail_len:]) < kTailInHeadThreshold * np.mean(I_energy[0:head_tail_len])\
                    and \
                np.mean(Q_energy[-head_tail_len:]) < kTailInHeadThreshold * np.mean(Q_energy[0:head_tail_len]):
                indexes_unbalance.append(i)
            process_bar.UpdateBar(i)
        # Get low power samples
        I_mean_energy = np.mean(I_energies)
        Q_mean_energy = np.mean(Q_energies)
        indexes_low_power = []
        sh_logger.debug("Detect Low Power")
        process_bar = ProcessBar(I_energies.shape[0] - 1)
        for i in range(I_energies.shape[0]):
            I_energy = I_energies[i, :]
            Q_energy = Q_energies[i, :]
            if np.mean(I_energy) < kSampleInBatchMeanThreshold * I_mean_energy\
                    or \
               np.mean(Q_energy) < kSampleInBatchMeanThreshold * Q_mean_energy   :
                indexes_low_power.append(i)
            process_bar.UpdateBar(i)

        half_tail_indexes = list(filter(lambda index: index not in indexes_low_power, indexes_unbalance))
        if kIsDebug:
            print("Plot half tail indexes")
            PlotSamples(I_data[half_tail_indexes, :])
            PlotSamples(Q_data[half_tail_indexes, :])

        head_half_tail_I = I_data[half_tail_indexes, :1000]
        tail_half_tail_I = I_data[half_tail_indexes, -1000:]
        head_half_tail_Q = Q_data[half_tail_indexes, :1000]
        tail_half_tail_Q = Q_data[half_tail_indexes, -1000:]

        head_mean_power = np.mean(head_half_tail_I ** 2 + head_half_tail_Q ** 2)
        tail_mean_power = np.mean(tail_half_tail_I ** 2 + tail_half_tail_Q ** 2)
        named_SNRs[module_name] = 10 * np.log10(head_mean_power / tail_mean_power)

        print(f'Current SNR is {10 * np.log10(head_mean_power / tail_mean_power)}')
    return named_SNRs


if __name__ == "__main__":
    named_SNRs = GetSNRs(kH5ModuleDataDir)
    SNRs = list(named_SNRs.values())
    mean_SNR = np.mean(SNRs)
    std_SNR = np.std(SNRs)
    with open(os.path.join(kLogFile, f'{os.path.split(kH5DataDir)[1]}_environment_noise_analysis.json'), 'w') as jf:
        json.dump({
            'named_SNRs': named_SNRs,
            'mean_SNR': mean_SNR,
            'std_SNR': std_SNR
        }, jf)
