#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: inceptionresnet_1D_train_val.py
@time: 2019/9/2 13:03
@desc:
"""
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from inceptionresnet_v2_model.inceptionresnet_1D import InceptionResNet1D
from inceptionresnet_v2_model.data_manager import DataManager
from inceptionresnet_v2_model.torch_saver import Saver
from inceptionresnet_v2_model.multi_classification_testor import MultiClassificationTester
from my_py_tools.my_logger import Logger
from my_py_tools.my_process_bar import ProcessBar
import my_py_tools.const as CNST
import time

# ! Manual Setting
kBatchSize = 25
kLearningRate = 0.001
kNumEpochs = 3
kSnapshotMaxToKeep = 20

kH5DataDir = os.path.join('..', 'data', 'clean_h5data.diff_module_same_mac_mini5')
kLogDirComment = ''
kTrainLogInterval = 10
kTestLogMultiplier = 5
kTestSamplesNum = 250

kIsRecover = False
kRecoverEpochID = 0

kIOnly = True
# ! Automatic Generated
kLogDir = os.path.join('.', 'log', f'torch.{os.path.split(kH5DataDir)[1]}.ICRS.{kLogDirComment}.log')
kTrainValLogFile = os.path.join(kLogDir, 'train_val.log')

kH5ModuleDataDir = os.path.join(kH5DataDir, 'h5_module_data')
kH5TrainTestDataDir = os.path.join(kH5DataDir, 'h5_train_test_split')
kSnapshotFileStr = os.path.join(kLogDir, 'snapshot', 'InceptionResNet1D-{}.snapshot')

kRecoverMetaFile = kSnapshotFileStr.format(kRecoverEpochID)
kRecoverDataFile = kSnapshotFileStr.format(kRecoverEpochID)

def TestSamples(samples, gts, net, tester):
    net.eval()
    sum_loss = 0
    i1 = 0
    while i1 < len(samples):
        if i1 + kBatchSize < len(samples):
            i2 = i1 + kBatchSize
        else:
            i2 = len(samples)
        batch_X = samples[i1:i2].reshape(i2 - i1, 1 if kIOnly else 2, -1)
        batch_X = torch.tensor(batch_X, dtype=torch.float32, device='cuda')
        cpu_batch_Y = gts[i1:i2]
        batch_Y = torch.tensor(cpu_batch_Y, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            loss, PR = net.get_cross_entropy_loss(batch_X, batch_Y, need_PR=True, is_expanded_target=True)
        sum_loss += loss * (i2 - i1)
        tester.update_confusion_matrix(PR.cpu().numpy(), cpu_batch_Y)
        i1 += kBatchSize
    tester.measure()
    net.train()
    return float(sum_loss) / len(samples), tester.micro_avg_precision


if __name__ == '__main__':
    # * data, log manager and saver, testor
    data_manager = DataManager(kH5TrainTestDataDir, kH5ModuleDataDir, I_only=kIOnly, down_sample=0)
    logger = Logger(kTrainValLogFile).logger
    writer = SummaryWriter(kLogDir)
    saver = Saver(kSnapshotFileStr)
    tester = MultiClassificationTester(data_manager.classes_list)
    # * build OR recover model
    net = InceptionResNet1D(data_manager.classes_num, num_input_channels= 1 if kIOnly else 2, batch_size=kBatchSize)
    dummy_input = torch.randn((1, net.input_size[1], net.input_size[2]), dtype=torch.float32)
    # writer.add_graph(net, (dummy_input, ), verbose=True)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=kLearningRate)
    if not kIsRecover:
        saver.save(epochID=-1, model=net, optimizer=optimizer)
    else:
        saver.load(epochID=kRecoverEpochID, model=net, optimizer=optimizer)
    # * Start training
    iteration = 0
    # ** If recover training, start at recover epoch
    if kIsRecover:
        start_epoch = kRecoverEpochID + 1
    else:
        start_epoch = 0

    for epochID in range(start_epoch, kNumEpochs):
        epoch_start_time = time.time()
        logger.info('****** Epoch: {}/{} ******'.format(epochID, kNumEpochs - 1))

        batches_num = int(np.ceil(data_manager.train_samples_num / kBatchSize))
        # * Init data_manager & Get batches generator
        data_manager.init_epoch()
        train_batches = data_manager.get_train_batches(kBatchSize)
        # * Init iteration
        sum_loss = 0
        # * Process bar
        process_bar = ProcessBar(batches_num)
        for i, train_batch in enumerate(train_batches):
            # * Process Batch Data
            batch_X, cpu_batch_Y = train_batch
            batch_X = batch_X.reshape(batch_X.shape[0], 1 if kIOnly else 2, -1)
            batch_X = torch.tensor(batch_X, dtype=torch.float32, device='cuda')
            batch_Y = torch.tensor(cpu_batch_Y, dtype=torch.float32, device='cuda')
            # Test every log_interval iteration
            if iteration % kTrainLogInterval == 0 and iteration != 0:
                train_loss = sum_loss / kTrainLogInterval
                tester.measure()
                train_accuracy = tester.micro_avg_precision
                writer.add_scalar('train/loss', train_loss, global_step=iteration)
                writer.add_scalar('train/accuracy', train_accuracy, global_step=iteration)
                # * Reset static loss
                sum_loss = 0
                # * Restart for test
                tester.restart()
                if (iteration / kTrainLogInterval) % kTestLogMultiplier == 0:
                    test_X, test_Y = data_manager.get_random_test_samples(kTestSamplesNum)
                    test_loss, test_accuracy = TestSamples(test_X, test_Y, net, tester)
                    writer.add_scalar('test/loss', test_loss, global_step=iteration)
                    writer.add_scalar('test/accuracy', test_accuracy, global_step=iteration)

                    process_bar.SkipMsg(
                        '({}/{}) train_loss: {}, train_accuracy: {}, test_loss: {}, test_accuracy: {}'.format(
                            i, batches_num - 1, train_loss, train_accuracy, test_loss, test_accuracy)
                        , logger)
                    # * Add compare scalar
                    writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, global_step=iteration)
                    writer.add_scalars('accuracy', {'train': train_accuracy, 'test': test_accuracy},
                                       global_step=iteration)
                    # * Restart tester for next interval
                    tester.restart()
                else:
                    process_bar.SkipMsg(
                        'Train: ({}/{}) loss: {}, accuracy: {}'.format(i, batches_num - 1, train_loss, train_accuracy)
                        , logger)
            # * Fwd, Bwd, Optimize, record
            optimizer.zero_grad()
            loss, PR = net.get_cross_entropy_loss(batch_X, batch_Y, is_expanded_target=True, need_PR=True)
            loss.backward()
            optimizer.step()
            tester.update_confusion_matrix(PR.cpu().numpy(), cpu_batch_Y)
            sum_loss += float(loss)
            # * Update bar and iteration
            iteration += 1
            process_bar.UpdateBar(i + 1)
        process_bar.Close()
        # ! Save model for this epoch
        saver.save(epochID=epochID, model=net, optimizer=optimizer,
                   )
        logger.info("It Cost {}s to finish this epoch".format(time.time() - epoch_start_time))
    writer.close()

