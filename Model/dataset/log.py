#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):#日志数据集的封装类，通过option的seq\qua\sem确定特征向量
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):#获取日志的各项信息
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
            #print("quantity:",self.Quantitatives)
        if self.sem:
            self.Semantics = logs['Semantics']
            #print("Sematic:",len(self.Semantics[0][0]))
        self.labels = labels

    def __len__(self):#返回日志序列数
        return len(self.labels)

    def __getitem__(self, idx):#迭代器 返回对应idx索引的一条日志序列的信息及标注
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            #print("idx",idx)
            #print("sema_idx",self.Semantics[idx])
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        return log, self.labels[idx]


if __name__ == '__main__':
    data_dir = '../../data/hdfs/hdfs_train'
    window_size = 10
    train_logs = prepare_log(data_dir=data_dir,
                             datatype='train',
                             window_size=window_size)
    train_dataset = log_dataset(log=train_logs, seq=True, quan=True)#获得训练的日志数据集
    print(train_dataset[0])
    print(train_dataset[100])
