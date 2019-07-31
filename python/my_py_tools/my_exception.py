#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: my_logger.py
@time: 2019/7/29 12:18
@desc:
"""


class MyError(Exception):

    def __init__(self, value = None, msg = 'An Error Occurred'):
        self.value = value
        self.msg = msg

    def __str__(self):
        return self.msg


class IllegalPhaseError(MyError):

    def __init__(self, msg = "Illegal Phase Error"):
        super(IllegalPhaseError, self).__init__(msg = msg)
