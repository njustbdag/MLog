
import tensorflow as tf



import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

from typing import *
from pathlib import Path
from enum import IntEnum

import torch.nn.functional as F
DATA_ROOT = Path("../data/brown")

N_EPOCHS = 210


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2




class MogLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz,cnn_length, in_size2=300,num_layers=2,mog_iteration=2,filter_num=3,filter_sizes="3,4,5",pool=True,max_length=100):
        super(MogLSTM, self).__init__()
        self.num_layers=num_layers
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.cnn_length=cnn_length
        self.mog_iterations = mog_iteration
        self.in_size2=in_size2

        self.liner_in=nn.Linear(self.input_size,self.in_size2)#输入维度降维至300


        self.lstm = nn.LSTM(input_sz,
                            hidden_sz,
                            num_layers=num_layers,
                            batch_first=True)
        # 这里hiddensz乘4，是将四个门的张量运算都合并到一个矩阵当中，后续再通过张量分块给每个门
        self.Wih = Parameter(torch.Tensor(self.in_size2, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))
        self.dropout=nn.Dropout(0.5)
        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz, self.in_size2))
        self.R = Parameter(torch.Tensor(self.in_size2, hidden_sz))
        self.fc = nn.Linear(self.hidden_size, 2)  # 隐藏层到输出层
        self.init_weights()
        self.max_length=max_length
        self.attn=nn.Linear(self.hidden_size*2,self.max_length)
        self.attn_combine=nn.Linear(self.hidden_size*2,self.hidden_size)

        self.w_omega = nn.Parameter(torch.Tensor(
            hidden_sz , hidden_sz ))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_sz , 1))
        self.query=nn.Parameter(torch.Tensor(
            hidden_sz , hidden_sz ))
        self.decoder = nn.Linear( hidden_sz, 2)
        self.w1 = nn.Parameter(torch.Tensor(
            hidden_sz, 1))
        self.f1 = nn.Sequential(nn.Linear((hidden_sz - self.cnn_length + 1), 2))
        self.fc2=nn.Linear(50*hidden_sz,2)
        self.fc3=nn.Linear(50-cnn_length+1,2)
        self.conv = nn.Conv1d(1, 1, cnn_length)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.w1, -0.1, 0.1)
        tf.compat.v1.disable_eager_execution()
        self.dropoutKeepProb = tf.compat.v1.placeholder(tf.float32, name="dropoutKeepProb")
        self.cnn=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(cnn_length,hidden_sz))


        self.filter_num=filter_num
        self.filter_sizes=[int(fsz) for fsz in filter_sizes.split(',')]
        self.convs=nn.ModuleList([nn.Conv2d(1,filter_num,(fsz,self.hidden_size)) for fsz in self.filter_sizes])
        self.linear=nn.Linear(len(self.filter_sizes)*filter_num,2)
    def init_weights(self):
        """
        权重初始化，对于W,Q,R使用xavier
        对于偏置b则使用0初始化
        :return:
        """
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)



    def mogrify(self, xt, ht):
        """
        计算mogrify
        :param xt:
        :param ht:
        :return:
        """
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R) * ht)
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q) * xt)
        return xt, ht

    def forward(self, features, device,init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x=features[0]
        #print("x",x.shape)
        #x = self.attention(x)
        batch_sz, seq_sz, _ = x.size()
        self.seq_size=seq_sz


        x=torch.tanh(self.liner_in(x))




        hidden_seq = []
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states



        for t in range(seq_sz):
            xt = x[:, t, :]
            #print("xt", xt.shape)
            #print("ht", ht.shape)
            xt, ht = self.mogrify(xt, ht)
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # chunk方法将tensor分块

            # LSTM
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            # outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)

            hidden_seq.append(ht.unsqueeze(Dim.batch))  # unsqueeze是给指定位置加上维数为1的维度
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        x=hidden_seq.cuda().view(-1, 1,hidden_seq.size(1) ,self.hidden_size)
        x=[F.relu(conv(x)) for conv in self.convs]
        #print("x",len(x),len(x[0]),len(x[0][0]),len(x[2][0][0]),len(x[0][0][0][0]))

        x=[F.max_pool2d(input=x_item,kernel_size=(x_item.size(2),x_item.size(3))) for x_item in x]
        #print("x_pooled", len(x), len(x[0]), len(x[0][0]), len(x[2][0][0]), len(x[0][0][0][0]))
        x=[x_item.view(x_item.size(0),-1) for x_item in x]
        x=torch.cat(x,1)
        x=self.dropout(x)
        out =self.linear(x)
        return out#, (ht, Ct)

