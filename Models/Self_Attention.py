# -*- coding: utf-8 -*-

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Self_Attention(nn.Module):
    def __init__(self, head):
        super(Self_Attention, self).__init__()
        self.num_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.num_baseline_nodes = hp.num_baseline_nodes
        self.head = head
        self.self_attention_nodes = hp.self_attention_nodes

        self.kernel1 = nn.Linear(self.num_baseline_nodes*2, self.self_attention_nodes)
        self.kernel2 = nn.ModuleList([nn.Linear(self.self_attention_nodes,1) for i in range(self.head)])

    def forward(self, x):
        attention_head = []
        for i in range(self.head):
            #print(self.kernel1(x).size())
            alpha = (F.softmax(self.kernel2[i](torch.tanh(self.kernel1(x))),dim=1))
            #print((alpha*x).sum(dim=1).size())
            if i == 0:
                attention_head = (alpha*x).sum(dim=1)
            else:
                attention_head = torch.cat((attention_head, (alpha*x).sum(dim=1)),1)
                #print((alpha*x).sum(dim=1))
        #print(attention_head.size())
        return attention_head
