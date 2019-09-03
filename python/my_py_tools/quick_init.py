#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: quick_init.py
@time: 2019/9/3 1:52
@desc:
"""
def QuickInit(obj, local_var_dict, update_to_obj_dict=True):
    init_dict_keys = obj.__init__.__code__.co_varnames[1:obj.__init__.__code__.co_argcount]
    init_dict = {key: local_var_dict[key] for key in init_dict_keys}
    if update_to_obj_dict:
        obj.__dict__.update(init_dict)
    return init_dict
