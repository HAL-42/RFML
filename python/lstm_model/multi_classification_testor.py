#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: multi_classification_testor.py
@time: 2019/8/6 2:20
@desc:
"""
import numpy as np
import pandas as pd
from typing import Optional
from my_py_tools.my_logger import sh_logger
import seaborn as sns
from matplotlib import pyplot as plt
import json


class MultiClassificationTester(object):
    def __init__(self, classes_list: list):
        self.classes_list = classes_list
        self.classes_num = len(classes_list)
        # ! confusion matrix attr
        self.confusion_matrix = np.zeros((self.classes_num, self.classes_num), dtype=np.int)
        self.pd_confusion_matrix = pd.DataFrame(data=self.confusion_matrix,
                                                index=classes_list, columns=classes_list)
        # ! attr need measured
        self.classes_gt_num = np.zeros(self.classes_num, dtype=np.int)
        self.classes_predict_num = np.zeros(self.classes_num, dtype=np.int)
        self.TP = self.confusion_matrix.diagonal()
        self.total_samples = 0
        self.classes_recall = np.zeros(self.classes_num, dtype=np.float)
        self.classes_precision = np.zeros(self.classes_num, dtype=np.float)
        self.classes_F1_score = np.zeros(self.classes_num, dtype=np.float)
        self.macro_avg_precision = 0
        self.macro_avg_F1_score = 0
        self.micro_avg_precision = 0
        self.attr_need_measured = {
            "Classes' Ground Truth Num": self.classes_gt_num, "Classes' Predict Num": self.classes_predict_num,
            "Classes' TP": self.TP, "Classes' Total Samples Num": self.total_samples,
            "Classes' Recall": self.classes_recall, "Classes' Precision": self.classes_precision,
            "Classes' F1 Score": self.classes_F1_score,
            "Classes' Macro F1 Score": self.macro_avg_F1_score, "Classes' Macro Precision": self.macro_avg_precision,
            "Classes' Micro Precision": self.micro_avg_precision,
        }
        # ! track measuring state
        self._is_measured = False

    def restart(self):
        self.confusion_matrix[...] = 0
        self.pd_confusion_matrix[...] = 0
        self._is_measured = False

    def update_confusion_matrix(self, samples_predict_vec: np.ndarray, samples_gt_vec: np.ndarray):
        samples_predict = np.argmax(samples_predict_vec, axis=1)
        samples_gt = np.argmax(samples_gt_vec, axis=1)

        for i in range(samples_predict.shape[0]):
            self.confusion_matrix[samples_gt[i], samples_predict[i]] += 1
            self.pd_confusion_matrix.iloc[samples_gt[i], samples_predict[i]] += 1
        self._is_measured = False

    def measure(self, weighted_macro_avg: bool = False, probability_weight: Optional[np.ndarray]=None):
        self.classes_gt_num = np.sum(self.confusion_matrix, axis=1)
        self.classes_predict_num = np.sum(self.confusion_matrix, axis=0)

        self.TP = self.confusion_matrix.diagonal()
        self.total_samples = np.sum(self.classes_gt_num)

        for i in range(self.classes_num):
            self.classes_recall[i] = self.TP[i] / self.classes_gt_num[i]
            self.classes_precision[i] = self.TP[i] / self.classes_predict_num[i]
            self.classes_F1_score[i] = \
                (2 * self.classes_precision[i] * self.classes_recall[i]) / \
                (self.classes_precision[i] + self.classes_recall[i])

        self.micro_avg_precision = np.sum(self.TP) / np.sum(self.classes_gt_num)
        if weighted_macro_avg == True and probability_weight != None:
            self.macro_avg_precision = np.sum(probability_weight * self.classes_precision)
            self.macro_avg_F1_score = np.sum(probability_weight * self.classes_precision)
        elif weighted_macro_avg == True and probability_weight == None:
            self.macro_avg_precision = \
                np.sum((self.classes_gt_num / self.total_samples) * self.classes_precision)
            self.macro_avg_F1_score = \
                np.sum((self.classes_gt_num / self.total_samples) * self.classes_precision)
        else:
            self.macro_avg_precision = np.sum((1 / self.classes_num) * self.classes_precision)
            self.macro_avg_F1_score = np.sum((1 / self.classes_num) * self.classes_precision)

        self._is_measured = True

    def show_confusion_matrix(self, img_save_path: Optional[str]=None):
        ax = sns.heatmap(self.pd_confusion_matrix,  # 指定绘图数据
                         # cmap=plt.cm.Blues,  # 指定填充色
                         linewidths=.1,  # 设置每个单元方块的间隔
                         annot=True  # 显示数值
                         )

        plt.xticks(np.arange(self.classes_num) + 0.5, self.classes_list)

        plt.yticks(np.arange(self.classes_num) + 0.5, self.classes_list)
        plt.yticks(rotation=0)

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predict')
        ax.set_ylabel('Ground Truth')

        # 显示图形
        fig = plt.gcf()
        plt.show()
        if img_save_path:
            fig.savefig(img_save_path)
        print("********Confusion Matric********")
        print(self.pd_confusion_matrix)

    def show_measure_result(self, rslt_save_path: Optional[str]=None):
        self._update_attr_need_measured()
        if not self._is_measured:
            sh_logger.warning("Using Unmeasured Attribution of MultiClassificationTester")
        sh_logger.info("Now Print Out Test Result")
        for key, value in self.attr_need_measured.items():
            print("**************************************")
            print(key)
            print(value)
        with open(rslt_save_path, 'w') as txt_f:
            for key, value in self.attr_need_measured.items():
                txt_f.write("**************************************\n")
                txt_f.write(key + "\n")
                txt_f.write(str(value) + "\n")

    def get_measure_result(self, index_name: Optional[str]=None):
        self._update_attr_need_measured()
        if not self._is_measured:
            sh_logger.warning("Using Unmeasured Attribution of MultiClassificationTester")
        if index_name:
            return self.attr_need_measured[index_name]
        else:
            return self.attr_need_measured

    def _update_attr_need_measured(self):
        self.attr_need_measured = {
            "Classes' Ground Truth Num": self.classes_gt_num, "Classes' Predict Num": self.classes_predict_num,
            "Classes' TP": self.TP, "Classes' Total Samples Num": self.total_samples,
            "Classes' Recall": self.classes_recall, "Classes' Precision": self.classes_precision,
            "Classes' F1 Score": self.classes_F1_score,
            "Classes' Macro F1 Score": self.macro_avg_F1_score, "Classes' Macro Precision": self.macro_avg_precision,
            "Classes' Micro Precision": self.micro_avg_precision,
        }