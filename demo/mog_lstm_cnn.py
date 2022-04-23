#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')


from Model.tools.predict import Predicter
from Model.tools.train import Trainer
from Model.tools.utils import *

from Model.models.mog_lstm_cnn2 import *
# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cuda"

# Smaple
options['sample'] = "session_window"
options['window_size'] = -1

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 768
options['hidden_size'] = 648
options['num_layers'] = 2
options['num_classes'] = 2
options['cnn_length']=3
# Train
options['batch_size'] = 32
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 150
options['lr_step'] = (40, 50)
options['lr_decay_ratio'] = 0.1

options['resume_path'] =None#"../result/mog_lstm_cnn/mog_lstm_cnn_last.pth"
options['model_name'] = "mog_lstm_cnn"
options['save_dir'] = "../result/mog_lstm_cnn/"
options['Event_TF_IDF']=1
options['template']="../data/HDFS/HDFS.log_templates.csv"
# Predict
options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_last.pth"
options['num_candidates'] = -1
options['data_type']='HDFS'
options['filter_num']=32
options['filter_size']='2,3,4'
options['pool']=True
seed_everything(seed=1234)
#input_size = 512
#hidden_size = 512
vocab_size = 2
batch_size = 4
lr = 3e-3
dropout = 0.5            # for simplicity: input dropout and output_dropout are 0.5. See appendix B in the paper for exact values
tie_weights = True       # in the paper, embedding weights and output weights are tied
betas = (0, 0.999)       # in the paper the momentum term in Adam is ignored
weight_decay = 2.5e-4    # weight decay is around this value, see appendix B in the paper
clip_norm = 10           # paper uses cip_norm of 10
batch_sz, seq_len, feat_sz, hidden_sz = 5, 10, 32, 16

def train():
    Model =MogLSTM(in_size2=300,input_sz=options['input_size'], hidden_sz=options['hidden_size'],cnn_length=options['cnn_length'],mog_iteration=2,filter_num=options['filter_num'],filter_sizes=options['filter_size'],pool=options['pool'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model =  MogLSTM(in_size2=300,input_sz=options['input_size'], hidden_sz=options['hidden_size'],cnn_length=options['cnn_length'],mog_iteration=2,filter_num=options['filter_num'],filter_sizes=options['filter_size'],pool=options['pool'])
    predicter = Predicter(Model, options)
    predicter.predict_supervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   # parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
   # if args.mode == 'train':+
    #
    #0
    train()

    predict()
   # else:
       # predict()