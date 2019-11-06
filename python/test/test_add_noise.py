#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: test_add_noise.py
@time: 2019/9/21 23:20
@desc:
"""
import numpy as np
import unittest
from noise_adder import NoiseAdder

kSamplesShape = (100, 10000)
kPSignal = 10
kAddSNR = 10
kOriginSNR = 10
kObjSNR = -10

class TestNoiseAdder(unittest.TestCase):

    def setUp(self) -> None:
        print("------Start an New Test-------")

    def tearDown(self) -> None:
        print("---------End an Test----------")

    def __init__(self, *args, **kwargs):
        super(TestNoiseAdder, self).__init__(*args, **kwargs)
        self.noise_adder = NoiseAdder()
        self.signals = np.random.randn(*kSamplesShape) * np.sqrt(kPSignal)

    @unittest.skip("Passed Already")
    def test_add_noise_by_SNR(self):
        """Test Add Noise By SNR"""
        noise_signals = self.noise_adder.add_noise_by_SNR(self.signals, kAddSNR)

        noises = noise_signals - self.signals
        P_signals = np.mean(self.signals ** 2, axis=1)
        P_noises = np.mean(noises ** 2, axis=1)
        add_SNRs = 10 * np.log10(P_signals / P_noises)

        SNR_rlat_errs = np.abs(add_SNRs - kAddSNR) / np.abs(kAddSNR)
        print(f'The Relative Error of SNR is {SNR_rlat_errs}')
        self.assertTrue(np.max(SNR_rlat_errs) < 0.02)

    def test_increase_noise_to_obj_SNR(self):
        """Test Increase Noise to Objective SNR"""
        origin_noise_signals = self.noise_adder.add_noise_by_SNR(self.signals, kOriginSNR)
        obj_noise_signals = self.noise_adder.increase_noise_to_obj_SNR(origin_noise_signals,
                                                                       kOriginSNR, kObjSNR)

        obj_noises = obj_noise_signals - self.signals
        P_obj_noises = np.mean(obj_noises ** 2, axis=1)
        P_signals = np.mean(self.signals ** 2, axis=1)
        obj_SNRs = 10 * np.log10(P_signals / P_obj_noises)

        SNR_rlat_errs = np.abs(obj_SNRs - kObjSNR) / np.abs(kObjSNR)
        print(f'The Relative Error of SNR is {SNR_rlat_errs}')
        self.assertTrue(np.max(SNR_rlat_errs) < 0.02)

if __name__ == '__main__':
    unittest.main(verbosity=2)