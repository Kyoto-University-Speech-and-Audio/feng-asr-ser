import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from Models.Self_Attention import Self_Attention
import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.num_decoder_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.num_baseline_nodes = hp.num_baseline_nodes
        # encoder
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim

        if hp.baseline_type == 'CNN_BLSTM':
            self.conv1 = nn.Conv2d(in_channels = 1, out_channels = hp.out_channels, kernel_size = 3,
                            stride = 1, padding = 1)
            self.conv2 = nn.Conv2d(in_channels = hp.out_channels, out_channels = hp.out_channels, kernel_size = 3,
                            stride = 1, padding = 1)
            self.maxp = nn.MaxPool2d(kernel_size = 3, stride = 3)

        if hp.baseline_type == 'CNN_BLSTM':
            self.lstm = nn.LSTM(input_size = input_size//3, hidden_size=self.num_baseline_nodes,
                                  num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size = input_size//3, hidden_size=self.num_baseline_nodes,
                                num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

        self.dense = nn.Linear(self.num_baseline_nodes*8*2, hp.num_baseline_nodes*2*2)
        #self.maxpool = nn.MaxPool1d(3, stride=2, ceil_mode = True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(self.num_baseline_nodes*2*2,hp.num_emotion)
        if hp.attention_type == 'Self_Attention':
            self.Self_Attention1 = Self_Attention(8)

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
            x = cnnout.reshape(hp.batch_size, -1, cnnout.size(3))
        if hp.baseline_type != 'CNN_BLSTM' and hp.baseline_type != "lim_BLSTM":
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        #self.lstm2.flatten_parameters()
        hbatch, _ = self.lstm(x)
        if hp.baseline_type != 'CNN_BLSTM' and hp.baseline_type != "lim_BLSTM":
            hbatch, lengths = nn.utils.rnn.pad_packed_sequence(hbatch, batch_first=True, total_length=total_length)
        #print(hbatch.size())
        if hp.attention_type == 'Self_Attention':
            maxp = self.Self_Attention1(hbatch)
        else:
            maxp=hbatch.mean(dim=1)
        #maxp = self.maxpool(hbatch)
        res = self.activation(self.dense(maxp))
        #print(res.size())
        emotion = self.output(res)
        #print(emotion.size())

        return emotion

    def decode(self, x, lengths):
        with torch.no_grad():
            total_length = x.size(1)
            if hp.baseline_type == 'CNN_BLSTM':
                x = torch.unsqueeze(x,1)
                cnnout = self.conv1(x)
                cnnout = F.relu(cnnout)
                cnnout = self.conv2(cnnout)
                cnnout = F.relu(cnnout)
                cnnout = self.maxp(cnnout)
                x = cnnout.reshape(1,-1,cnnout.size(3))
            #x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            hbatch, _ = self.lstm(x)
            #hbatch, lengths = nn.utils.rnn.pad_packed_sequence(hbatch, batch_first=True, total_length=total_length)
            #print(hbatch.size())
            #maxp = torch.zeros((hp.batch_size, hp.num_baseline_nodes), dtype=torch.float32, device=DEVICE)
            #print(hbatch.size())
            if hp.attention_type == 'Self_Attention':
                maxp = self.Self_Attention1(hbatch)
            else:
                maxp=hbatch.mean(dim=1)
            #maxp = self.maxpool(hbatch)
            res = self.activation(self.dense(maxp))
            #print(res.size())
            emotion = self.output(res)
            #print(emotion.size())

        emotion = emotion.squeeze()
        if hp.score_func == 'log_softmax':
            emotion = F.log_softmax(emotion, dim=0)
        elif hp.score_func == 'softmax':
            emotion = F.softmax(emotion, dim=0)

        #import pdb;pdb.set_trace()
        bestidx = emotion.data.argmax().item()

        return bestidx
