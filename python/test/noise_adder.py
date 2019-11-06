#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: noise_adder.py
@time: 2019/9/21 23:36
@desc:
"""
import numpy as np

class NoiseAdder(object):

    def add_noise_by_SNR(self, signals, SNR):
        P_signal = np.mean(signals ** 2, axis=1)
        P_noise = P_signal * np.power(10, - SNR / 10)
        return signals + np.random.randn(*signals.shape) * np.expand_dims(np.sqrt(P_noise), 1)

    def increase_noise_to_obj_SNR(self, signals, origin_SNR, obj_SNR):
        incre_SNR = 10 * np.log10((np.power(10, origin_SNR / 10) + 1) /
                                  (np.power(10, (origin_SNR - obj_SNR) / 10) - 1))
        return self.add_noise_by_SNR(signals, incre_SNR)
