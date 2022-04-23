#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
sys.path.append('../../')
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.dataset.log import log_dataset
from Model.dataset.sample import session_window, read_json,sliding_window
from Model.tools.utils import (save_parameters, seed_everything,
                               train_val_split)

#将测试数据分成int数字，（并-1），然后把不足window_size的序列补-1
def generate(name):
    window_size = 10
    hdfs = {}#字典
    length = 0
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n , map(int, ln.strip().split())))#将文件每行分裂成int数字，(再-1)
            ln = ln + [-1] * (window_size + 1 - len(ln))#长度不足window_size的补-1
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1#重复的序列 让字典+1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.data_type=options['data_type']
        self.Event_TF_IDF = options['Event_TF_IDF']
        self.template = options['template']

    def read_json(filename):
        with open(filename, 'r') as load_f:
            file_dict = json.load(load_f)
        return file_dict



    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        predict_list=[]
        label_list=[]
        print('model_path: {}'.format(self.model_path))
        if self.data_type=='HDFS':
            test_logs, test_labels = session_window(self.data_dir, datatype='test',Event_TF_IDF=self.Event_TF_IDF,template=self.template)
        elif self.data_type=='BGL':
            test_logs, test_labels = session_window(self.data_dir, datatype='test',data_type='BGL',Event_TF_IDF=self.Event_TF_IDF,template=self.template)
        elif self.data_type=='OpenStack':
            test_logs, test_labels = session_window(self.data_dir, datatype='test', data_type='OpenStack',Event_TF_IDF=self.Event_TF_IDF,
                                                    template=self.template)
        elif self.data_type=='Thunderbird':
            test_logs, test_labels = session_window(self.data_dir, datatype='test', data_type='Thunderbird',Event_TF_IDF=self.Event_TF_IDF,
                                                    template=self.template)
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            #print("output:",output)

            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            #print("output:",output)
            predicted = (output < 0.2).astype(int)#BGL 0.5 HDFS:0.02 0.2



            label = np.array([y.cpu() for y in label])
            label_list.extend(label)
            #print("pred:",predicted.tolist())
            #print("label",label.tolist())

            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        print("TP:",TP,"FP",FP)
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {},true positive (TP): {},true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN,TP,TN, P, R, F1))

