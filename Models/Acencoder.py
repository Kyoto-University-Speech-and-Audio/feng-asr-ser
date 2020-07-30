# -*- coding: utf-8 -*-

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Acencoder(nn.Module):
    def __init__(self):
        super(Acencoder, self).__init__()
        self.num_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.num_baseline_nodes = hp.num_baseline_nodes
        #self.batch_size = hp.batch_size
        self.batch_size = 1

        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim
        if hp.baseline_type == 'CNN_BLSTM':
            self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3,
                            stride = 1, padding = 1)
            self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3,
                            stride = 1, padding = 1)
            self.maxp = nn.MaxPool2d(kernel_size = 3, stride = 3)

        if hp.baseline_type == 'CNN_BLSTM':
            self.lstm = nn.LSTM(input_size = input_size//3, hidden_size=self.num_hidden_nodes,
                                  num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size = input_size//3,
                                hidden_size=self.num_baseline_nodes,
                                num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
            #self.dense = nn.Linear(input_size, self.num_hidden_nodes*2)

    def forward(self, x, lengths):
        total_length = x.size(1)

        if hp.baseline_type == 'CNN_BLSTM':
            x = torch.unsqueeze(x, 1)
            cnnout = self.conv1(x)
            cnnout = F.relu(cnnout)
            cnnout = self.conv2(cnnout)
            cnnout = F.relu(cnnout)

            #print(cnnout.data.size())

            cnnout = self.maxp(cnnout)
            #print(cnnout.data.size())
            x = cnnout.reshape(self.batch_size, -1, cnnout.size(3))
            hbatch, _ = self.lstm(x)
        elif hp.baseline_type == 'lim_BLSTM':
            hbatch, _ = self.lstm(x)
        else:
            #hbatch = self.dense(x)
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            h, _ = self.lstm(x)
            hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        #print(hbatch.size())
        return hbatch
