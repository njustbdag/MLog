import torch
import torch.nn as nn
import math

class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r
   
    def mogrify(self, x, h):
        x=x.cuda()
        h=h.cuda()
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0: 
                h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
                #h=h.cuda()
            else:
                x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
                #x=x.cuda()
        return x, h

    def forward(self, x, states):
        x=x.cuda()

        ht, ct = states
        ht=ht.cuda()
        ct=ct.cuda()
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct
