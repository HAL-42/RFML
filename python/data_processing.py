#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: data_processing.py
@time: 2019/7/15 10:31
@desc:
"""

import glob
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import h5py
from typing import Optional, Union
import re
from my_py_tools.my_logger import Logger, sh_logger
from my_py_tools import const
from typing import Optional, Union
import json
from my_py_tools.my_process_bar import ProcessBar


kDataPath = os.path.join('..', 'data', 'data.diff_mac')
kH5DataPath = os.path.join('..', 'data', 'h5data.diff_mac')
kLogPath = os.path.join('.', 'log')
if not os.path.isdir(kLogPath):
    os.mkdir(kLogPath)

kCSVHeaderLen = 10
kSampleLen = 10000

kIsDebug = False
kIsTiming = True
kTrunkNaive = False

kNStep = 200

kTrainRatio = 0.8
kSaveStep = 50000


def TrunkQuiteTo10k(data: np.ndarray) -> Union[np.ndarray, int]:
    normalized_data = data / np.max(abs(data))
    if kTrunkNaive:
        # Most Naive Way:
        trunked_data = normalized_data[-kSampleLen - 1:-1]
        return trunked_data
    else:
        st_energy = []
        i1 = 0
        i2 = i1 + kNStep
        while i2 <= len(normalized_data):
            # Sadly use abs won't make this program quicker
            # st_energy.append(np.sum(np.abs(normalized_data[i1:i2])))
            st_energy.append(np.sum(normalized_data[i1:i2] ** 2))
            i1 = i2
            i2 += kNStep
        threshold = np.mean(st_energy)
        for i in range(len(st_energy)):
            if st_energy[i] > threshold:
                break
        start_point = i * kNStep + kNStep

        print("Start Point is at", start_point)
        print("Threshold =", threshold)
        if kIsDebug:
            plt.plot(np.array(st_energy))
            plt.show()
            print("Press Any Key to Continue:")
            input()
        if start_point + kSampleLen <= len(normalized_data):
            return normalized_data[start_point:start_point + kSampleLen]
        else:
            return start_point


def Get10kIQFromCSV(csv_path: str) -> Optional[tuple]:
    str_data_with_header = np.loadtxt(csv_path, dtype=str)
    str_data = str_data_with_header[kCSVHeaderLen:]
    I = []
    Q = []

    for one_line in str_data:
        str_I, str_Q = one_line.split(',')
        I.append(float(str_I))
        Q.append(float(str_Q))
    I = np.array(I, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    trunked_I = TrunkQuiteTo10k(I)
    trunked_Q = TrunkQuiteTo10k(Q)
    if kIsDebug:
        plt.subplot(2, 1, 1)
        plt.plot(I)
        plt.subplot(2, 1, 2)
        plt.plot(trunked_I)
        plt.show()
        print("Press Any Key to Continue:")
        input()

        plt.subplot(2, 1, 1)
        plt.plot(Q)
        plt.subplot(2, 1, 2)
        plt.plot(trunked_Q)
        plt.show()
        print("Press Any Key to Continue:")
        input()
    return trunked_I, trunked_Q


def SaveIQ(I_path: str, I: Union[np.ndarray, int], Q_path, Q: Union[np.ndarray, int]) -> int:
    if isinstance(I, int) or isinstance(Q, int):
        error_path = os.path.join(os.path.split(I_path)[0], 'error.txt')
        csv_name = os.path.split(I_path)[1].split('_')[0]
        with open(error_path, 'a') as err_f:
            err_f.write("{csv_name}'s I Start Point is at {I}, Q Start Point is at {Q}.\n"
                        .format(csv_name=csv_name, I=I, Q=Q))
    else:
        np.savetxt(I_path, I)
        np.savetxt(Q_path, Q)
    return 0


def DataProcess():
    # Process Each Dir
    for wifi_module_path in glob.glob(os.path.join(kDataPath, '*')):
        if os.path.isdir(wifi_module_path):
            print('-------------------------------------')
            print('Processing Data In', wifi_module_path)
            wifi_module_name = os.path.split(wifi_module_path)[1]

            if os.path.isdir(os.path.join(wifi_module_path, 'output.finished')):  # If finished, Skip
                sh_logger.warning("Current Wifi Module has been Processed. Skip")
                continue
            if not os.path.isdir(os.path.join(wifi_module_path, 'output')):  # Make output dir
                os.mkdir(os.path.join(wifi_module_path, 'output'))

            if kIsTiming:  # Time for Processing one folder
                start_folder_time = time.time()
                # Process Each CSV
            for csv_path in glob.glob(os.path.join(wifi_module_path, '*.csv')):
                csv_name = (os.path.split(csv_path)[1])[
                    :-4]  # Get csv name without '.csv'
                print('Processing CSV', csv_name,
                      'from wifi module', wifi_module_name)

                if kIsTiming:
                    start_csv_time = time.time()  # Time for processing one csv file
                # Prepare file path. If I/Q files exit, skip
                I_path = os.path.join(
                    wifi_module_path, 'output', csv_name + '_I' + '.txt')
                Q_path = os.path.join(
                    wifi_module_path, 'output', csv_name + '_Q' + '.txt')
                if os.path.isfile(I_path) and os.path.isfile(Q_path):
                    sh_logger.warning('Trunked data of ' +
                                    csv_name + ' already exits. Skip')
                else:
                    # Get Trunked and normalized IQ, save as txt
                    I, Q = Get10kIQFromCSV(csv_path)
                    SaveIQ(I_path, I, Q_path, Q)
                # Timing
                if kIsTiming:
                    print("Processing Current CSV cost", str(
                        time.time() - start_csv_time) + 's')
            if kIsTiming:
                print("Processing Current module cost", str(
                    time.time() - start_folder_time) + 's')
            # Label Current Wifi module as Finished
            os.rename(os.path.join(wifi_module_path, 'output'),
                      os.path.join(wifi_module_path, 'output.finished'))


def GetH5ModuleData(data_path: str, h5_module_data_path: str):
    logger = Logger(os.path.join(kLogPath, 'tran_to_h5_log.txt')).logger
    h5_module_data_txt_path = os.path.join(
        h5_module_data_path, 'h5_module_data.txt')

    # ! Process each wifi module
    module_data_num = dict()
    for wifi_module_path in glob.glob(os.path.join(data_path, '*')):
        if os.path.isdir(wifi_module_path):
            logger.info('-------------------------------------')
            logger.info('Processing Data In' + wifi_module_path)
            module_name = os.path.split(wifi_module_path)[1]
            # ! Convert only if output dir exits
            assert os.path.isdir(os.path.join(wifi_module_path, 'output.finished')), \
                "No output finished in {}".format(wifi_module_path)
            # ! Get I and Q paths and sort to make sure they are correspondent
            I_paths = []
            Q_paths = []
            for txt_file_path in glob.glob(os.path.join(wifi_module_path, 'output.finished', '*')):
                if re.search('_I', os.path.split(txt_file_path)[1]):
                    I_paths.append(txt_file_path)
                elif re.search('_Q', os.path.split(txt_file_path)[1]):
                    Q_paths.append(txt_file_path)
            I_paths.sort()
            Q_paths.sort()
            # ! Update module_data_num
            assert len(I_paths) == len(
                Q_paths), "Data Num of num_I != num_Q ! in {}".format(wifi_module_path)
            module_data_num[module_name] = len(I_paths)
            # ! Get Numpy and Write to H5
            I_data = np.empty((len(I_paths), kSampleLen), dtype=np.float32)
            Q_data = np.empty((len(Q_paths), kSampleLen), dtype=np.float32)
            labels = np.array([module_name] * len(I_paths))
            process_bar = ProcessBar(len(I_paths))
            for i in range(len(I_paths)):
                I_data[i, :] = np.loadtxt(I_paths[i], dtype=np.float32)
                Q_data[i, :] = np.loadtxt(Q_paths[i], dtype=np.float32)
                process_bar.UpdateBar(i + 1)
            process_bar.Close()
            with h5py.File(os.path.join(h5_module_data_path, module_name+'.h5'), 'w') as hf:
                hf.create_dataset("I", data=I_data)
                hf.create_dataset("Q", data=Q_data)
                # http://docs.h5py.org/en/stable/strings.html Store Strings in h5py
                dt = h5py.special_dtype(vlen=str)
                dset_label = hf.create_dataset("Label", labels.shape, dtype=dt)
                dset_label[:] = labels
            # ! Record in txt
            with open(h5_module_data_txt_path, 'a') as txt_f:
                txt_f.write("{} {}\n".format(
                    module_name, module_data_num[module_name]))
    return module_data_num


def _List2H5(sample_list: list , file_prefix:str ,h5_module_data_path: str,
             h5_train_test_data_path: str, logger) -> int:
    # ! Set basic info for prefix_info.json
    sample_num = len(sample_list)
    h5_names = []
    # ! Save Data Step by Step
    i1 = 0
    while i1 < sample_num:
        # i1 - i2 -----> prefix_i1-i2.h5
        if i1 + kSaveStep < sample_num:
            i2 = i1 + kSaveStep
        else:
            i2 = sample_num
        logger.info("In List {}, Processing Samples from {} to {}".format(file_prefix, i1, i2))
        # ! Get I, Q, Labels Data
        I_data = np.empty((i2 - i1, kSampleLen), dtype=np.float32)
        Q_data = np.empty((i2 - i1, kSampleLen), dtype=np.float32)
        labels = np.empty((i2 - i1,), dtype=np.object)
        process_bar = ProcessBar(i2 - i1)
        for i, sample in enumerate(sample_list[i1:i2]):
            with h5py.File(os.path.join(h5_module_data_path, sample[0]+'.h5'), 'r') as hf:
                I_data[i, :] = hf['I'][sample[1], :]
                Q_data[i, :] = hf['Q'][sample[1], :]
            labels[i] = sample[0]
            process_bar.UpdateBar(i + 1)
        # ! Save To h5 file
        h5_name = "{}_{}-{}.h5".format(file_prefix, i1, i2)
        h5_names.append(h5_name)
        h5_path = os.path.join(h5_train_test_data_path, h5_name)
        logger.info("Saving {}".format(file_prefix, h5_name))
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('I', data=I_data)
            hf.create_dataset('Q', data=Q_data)
            dset_labels = hf.create_dataset('Label', labels.shape, dtype=h5py.special_dtype(vlen=str))
            dset_labels[:] = labels
        i1 = i2
    with open(os.path.join(h5_train_test_data_path, file_prefix+"_info.json"), 'w') as jf:
        json.dump({"sample_num": sample_num, "save_step": kSaveStep, "h5_names": h5_names},
                  jf)
    return 0


def GetH5TrainTestData(h5_module_data_path: str, h5_train_test_data_path: str,
                       module_data_num: Optional[dict] = None) -> int:
    # ! Logger
    logger = Logger(os.path.join(kLogPath, 'train_test_data_log.txt')).logger
    # ! Get module_data_num
    if not module_data_num:
        module_data_num = dict()
        with open(os.path.join(h5_module_data_path, 'h5_module_data.txt')) as txt_f:
            lines = txt_f.readlines()
        for line in lines:
            match_rslt = re.match('(\S+) (\d+)', line)
            module_data_num[match_rslt.group(1)] = int(match_rslt.group(2))
    # ! Get sample, train, test num of per module
    sample_num = min([data_num for data_num in module_data_num.values()])
    train_num = int(sample_num * kTrainRatio)
    test_num = sample_num - train_num
    logger.info("For Each Module, sample num = {}, train_num = {}, test_num = {}".
                format(sample_num, train_num, test_num))
    # ! Get Train/Test list in the form like: [('A', 1), ('B', 2566), ..., ('E', 4563)]
    train_list = []
    test_list = []
    for module_name, data_num in module_data_num.items():
        shuffled_index = np.arange(0, data_num)
        np.random.shuffle(shuffled_index)
        train_list += [(module_name, index) for index in shuffled_index[0:train_num]]
        test_list += [(module_name, index) for index in shuffled_index[train_num:train_num + test_num]]
    np.random.shuffle(train_list)
    np.random.shuffle(test_list)
    # ! Save Train and Test Set to h5
    logger.info("Start Saving Train List")
    _List2H5(train_list, 'train', h5_module_data_path, h5_train_test_data_path, logger)
    logger.info("Start Saving Test List")
    _List2H5(test_list, 'test', h5_module_data_path, h5_train_test_data_path, logger)
    return 0


if __name__ == '__main__':
    DataProcess()

    h5_module_data_path = os.path.join(kH5DataPath, 'h5_module_data')
    h5_train_test_data_path = os.path.join(kH5DataPath, 'h5_train_test_split')

    if not os.path.isdir(kH5DataPath):
        os.mkdir(kH5DataPath)
    else:
        sh_logger.warning("Dir {} is Already Exits!".format(kH5DataPath))

    module_data_num = None
    if not os.path.isdir(h5_module_data_path):
        os.mkdir(h5_module_data_path)
        module_data_num = GetH5ModuleData(kDataPath, h5_module_data_path)
    else:
        sh_logger.warning("Dir {} is Already Exits. Skip Func GetH5ModuleData".
                                format(h5_module_data_path))
    if not os.path.isdir(h5_train_test_data_path):
        os.mkdir(h5_train_test_data_path)
        GetH5TrainTestData(h5_module_data_path,
                           h5_train_test_data_path, module_data_num)
    else:
        sh_logger.warning("Dir {} is Already Exits. Skip Func GetTrainH5TestData".
                                format(h5_train_test_data_path))