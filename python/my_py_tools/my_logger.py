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

import logging
from logging import handlers
import sys

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename=None,level='info',when=None,backCount=3,fmt='%(asctime)s - %(name)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        if not filename:
            self.logger = logging.getLogger('sh_logger')
        else:
            self.logger = logging.getLogger(filename)
        file_format_str = logging.Formatter(fmt)#设置日志格式
        sh_format_str = logging.Formatter('%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s')
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别

        sh = logging.StreamHandler(sys.stdout)#往屏幕上输出
        sh.setFormatter(sh_format_str) #设置屏幕上显示的格式
        self.logger.addHandler(sh) #把对象加到logger里

        if filename:
            if when:
                th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                                       encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
            else:
                th = handlers.logging.FileHandler(filename=filename, encoding='utf-8')
            # 实例化TimedRotatingFileHandler
            # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
            # S 秒
            # M 分
            # H 小时、
            # D 天、
            # W 每星期（interval==0时代表星期一）
            # midnight 每天凌晨
            th.setFormatter(file_format_str)  # 设置文件里写入的格式
            self.logger.addHandler(th)


sh_logger = Logger().logger


if __name__ == '__main__':
    log = Logger('all.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')

    sh_logger.warning("sh_waring")