'''
@Date: 2019-09-04 00:56:25
@Author: Xiaobo Yang
@Email: hal_42@zju.edu.cn
@Company: Zhejiang University
@LastEditors: Xiaobo Yang
@LastEditTime: 2019-09-04 01:00:55
@Description: 
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from inceptionresnet_v2_model.inceptionresnet_1D import InceptionResNet1D
from utils.data_manager import DataManager
from inceptionresnet_v2_model.torch_saver import Saver
from utils.multi_classification_testor import MultiClassificationTester
from my_py_tools.my_logger import Logger
from my_py_tools.my_process_bar import ProcessBar
from inceptionresnet_1D_test_model import TestSamples

from my_py_tools.const import Const
K = Const()

# ! Manual Setting
# * Path Setting
K.H5DataDir = os.path.join('..', 'data', 'clean_h5data.diff_module_same_mac_43')
K.LogDirComment = 'V1-B27-lre-3'
# * Model ,Optimizer and Recover Setting
K.IsRecover = False
K.LoadModelNum = 7200
# ** Model Init Setting
K.ModelSettings = {
    'num_incept_A': 5, 'num_incept_B': 10, 'num_incept_C': 5,
    'scale_A': 0.17, 'scale_B': 0.1, 'scale_C': 0.2
}
# * Training Setting
K.BatchSize = 27
K.NumEpochs = 5
# * Log, test and Snapshot Setting
K.TrainLogInterval = 10
K.TestLogMultiplier = 30
K.SnapshotMultiplier = 2
K.TestSamplesNum = 2500
K.TestBatchSize = 100
# ** Optimizer Setting
K.LearningRate = 0.001
# - Set None for no Exponential LR(About go down to 0.0005 for 1 epoch)
# K.ExponentialLR = np.power(0.00005 / 0.045, 1 / (16000 / K.TrainLogInterval * K.SnapshotMultiplier * K.SnapshotMultiplier))
K.ExponentialLR = None
# * Other Setting: Should Use both I+Q or just use I to train
K.IOnly = True
# ! Automatic Generated Setting
K.LogDir = os.path.join('.', 'log', f'torch.{os.path.split(K.H5DataDir)[1]}.ICRS.{K.LogDirComment}.log')
K.SummaryDir = os.path.join(K.LogDir, 'summary')
K.TrainValLogFile = os.path.join(K.LogDir, 'train_val.log')
K.SnapshotFileStr = os.path.join(K.LogDir, 'snapshot', 'InceptionResNet1D-{}.snapshot')

K.H5ModuleDataDir = os.path.join(K.H5DataDir, 'h5_module_data')
K.H5TrainTestDataDir = os.path.join(K.H5DataDir, 'h5_train_test_split')


if __name__ == '__main__':
    # * data, log manager and saver, tester
    data_manager = DataManager(K.H5TrainTestDataDir, K.H5ModuleDataDir, I_only=K.IOnly, down_sample=0)
    logger = Logger(K.TrainValLogFile).logger
    writer = SummaryWriter(K.SummaryDir)
    saver = Saver(K.SnapshotFileStr)
    tester = MultiClassificationTester(data_manager.classes_list)
    # * build OR recover model, optimizer
    # writer.add_graph(net, (dummy_input, ), verbose=True)
    if not K.IsRecover:
        model_init_dict = K.ModelSettings.update({
                                                'num_input_channels': 1 if K.IOnly else 2,
                                                'batch_size': K.BatchSize
                                                })
        net = InceptionResNet1D(data_manager.classes_num, **K.ModelSettings)
        # dummy_input = torch.randn((1, net.input_size[1], net.input_size[2]), dtype=torch.float32)
        optimizer = torch.optim.Adam(net.parameters(), lr=K.LearningRate)
        if K.ExponentialLR != None:
            exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, K.ExponentialLR, last_epoch=-1)
        # saver.save(model_num=-1, model=net, optimizer=optimizer)
        iteration = 0
        net.cuda()
    else:
        net, optimizer = saver.restore(K.LoadModelNum,model_cls=InceptionResNet1D,
                                       optimizer_cls=torch.optim.Adam, device='cuda')
        if K.ExponentialLR != None:
            exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, K.ExponentialLR, last_epoch=K.LoadModelNum)
        # ** If recover training, start at recover iteration
        iteration = K.LoadModelNum
    # * Start training
    # ** Restore epochID
    batches_num = int(np.ceil(data_manager.train_samples_num / K.BatchSize))
    start_epoch = int(np.round(iteration / batches_num))
    for epochID in range(start_epoch, K.NumEpochs):
        logger.info('****** Epoch: {}/{} ******'.format(epochID, K.NumEpochs - 1))

        # * Init data_manager & Get batches generator
        data_manager.init_epoch()
        train_batches = data_manager.get_train_batches(K.BatchSize)
        # * Init iteration, sum_loss and tester
        sum_loss = 0
        tester.restart()
        # * Process bar
        process_bar = ProcessBar(batches_num)
        for batch_ID, train_batch in enumerate(train_batches):
            # ! Test every log_interval iteration
            if iteration % K.TrainLogInterval == 0 and iteration != 0 and \
                    (not (K.IsRecover and iteration == K.LoadModelNum)):
                train_loss = sum_loss / K.TrainLogInterval
                tester.measure()
                train_accuracy = tester.micro_avg_precision
                writer.add_scalar('train/loss', train_loss, global_step=iteration)
                writer.add_scalar('train/accuracy', train_accuracy, global_step=iteration)
                writer.add_scalars('loss', {'train': train_loss}, global_step=iteration)
                writer.add_scalars('accuracy', {'train': train_accuracy}, global_step=iteration)
                # * Reset static loss and tester
                sum_loss = 0
                tester.restart()
                # * Output
                process_bar.SkipMsg(
                    'Train: ({}/{}) loss: {}, accuracy: {}'.format(batch_ID, batches_num - 1, train_loss,
                                                                   train_accuracy), logger)
            # ! If comes to test iteration, test part of test set samples
            if iteration % (K.TrainLogInterval * K.TestLogMultiplier) == 0:
                process_bar.SkipMsg('/*******Now Test the Model*******/', logger)
                # * Get test data
                test_X, test_Y = data_manager.get_random_test_samples(K.TestSamplesNum)
                # Test in eval+no_grad mode
                with torch.no_grad():
                    net.eval()
                    test_loss, test_accuracy = TestSamples(test_X, test_Y, net, tester, batch_size=K.TestBatchSize)
                net.train()
                writer.add_scalar('test/loss', test_loss, global_step=iteration)
                writer.add_scalar('test/accuracy', test_accuracy, global_step=iteration)
                process_bar.SkipMsg(
                    'Test: test_loss: {}, test_accuracy: {}'.format(test_loss, test_accuracy), logger)
                # * Add compare scalar
                writer.add_scalars('loss', {'test': test_loss}, global_step=iteration)
                writer.add_scalars('accuracy', {'test': test_accuracy},global_step=iteration)
                # * Restart tester for next interval
                tester.restart()
            # ! If time for take snapshot
            if iteration % (K.TrainLogInterval * K.TestLogMultiplier * K.SnapshotMultiplier) == 0 and \
                    not (K.IsRecover and (iteration == K.LoadModelNum)) :
                process_bar.SkipMsg(
                    f"Taking Snapshot of current model as {os.path.split(K.SnapshotFileStr.format(iteration))[1]}",
                                    logger)
                saver.save(iteration, net, optimizer, model_init_dict=net.model_init_dict)
                # ! Also decrease the lr here
                if K.ExponentialLR != None:
                    exp_scheduler.step()
                    process_bar.SkipMsg(f"Current lr is {exp_scheduler.get_lr()}", logger)
            # * Process Batch Data
            batch_X, cpu_batch_Y = train_batch
            batch_X = batch_X.reshape(batch_X.shape[0], 1 if K.IOnly else 2, -1)
            batch_X = torch.tensor(batch_X, dtype=torch.float32, device='cuda')
            batch_Y = torch.tensor(cpu_batch_Y, dtype=torch.float32, device='cuda')
            # * Fwd, Bwd, Optimize, record
            optimizer.zero_grad()
            loss, PR = net.get_cross_entropy_loss(batch_X, batch_Y, is_expanded_target=True, need_PR=True)
            loss.backward()
            optimizer.step()
            tester.update_confusion_matrix(PR.cpu().numpy(), cpu_batch_Y)
            sum_loss += float(loss)
            # * Update bar and iteration
            process_bar.UpdateBar(batch_ID + 1)
            iteration += 1
        process_bar.Close()
        # ! Save model for this epoch
        # saver.save(model_num=epochID, model=net, optimizer=optimizer,
        #            model_init_dict=net.model_init_dict)
    writer.close()