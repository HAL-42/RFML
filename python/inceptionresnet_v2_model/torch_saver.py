#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: torch_saver.py
@time: 2019/9/2 13:20
@desc:
"""
import torch
import os
from my_py_tools.my_logger import sh_logger


class Saver(object):

    def __init__(self, snapshot_file_str):
        self.snapshot_file_str = snapshot_file_str
        snapshot_dir = os.path.split(snapshot_file_str)[0]
        if not os.path.isdir(snapshot_dir):
            os.makedirs(snapshot_dir)
            sh_logger.info(f"Make snapshot dir at {snapshot_dir}")

    def save(self, epochID, model, optimizer, **append_dict):
        save_dict = {
            'epochID': epochID,
            'model_state_dict': model.state_dict,
            'optimizer_state_dict': optimizer.state_dict,
        }
        save_dict.update(append_dict)
        torch.save(save_dict, self.snapshot_file_str.format(epochID))

    def load(self, epochID, model=None, optimizer=None, device='cuda'):
        assert os.path.isfile(self.snapshot_file_str.format(epochID)), \
            f"No snapshot file called {self.snapshot_file_str.format(epochID)}"
        if device != 'cpu':
            save_dict = torch.load(self.snapshot_file_str.format(epochID))
        else:
            save_dict = torch.load(self.snapshot_file_str.format(epochID), map_location='cpu')

        if model != None:
            model.load_state_dict(save_dict['model_state_dict'], strict=False)
        if optimizer != None:
            optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        return save_dict

    def restore(self, epochID, model_cls=None, optimizer_cls=None, device='cuda'):
        assert os.path.isfile(self.snapshot_file_str.format(epochID)), \
            f"No snapshot file called {self.snapshot_file_str.format(epochID)}"
        if device != 'cpu':
            save_dict = torch.load(self.snapshot_file_str.format(epochID))
        else:
            save_dict = torch.load(self.snapshot_file_str.format(epochID), map_location='cpu')

        # * Restore model
        if model_cls != None:
            model = model_cls(**save_dict.get('model_init_dict', {}))
            model.load_state_dict(save_dict.get('model_state_dict', {}))
        else:
            model = None
        # * Retore optimizer
        if optimizer_cls != None:
            optimizer = optimizer_cls()
            optimizer.load_state_dict(save_dict.get('optimizer_state_dict', {}))
        else:
            optimizer = None

        return model, optimizer
