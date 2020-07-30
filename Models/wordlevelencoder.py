# -*- coding: utf-8 -*-
import copy
import numpy as np
from operator import itemgetter, attrgetter
import torch
import torch.nn.functional as F
import torch.nn as nn

from Models.Self_Attention import Self_Attention
import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Wordlevelencoder(nn.Module):
    def __init__(self):
        super(Wordlevelencoder, self).__init__()
        self.num_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.num_baseline_nodes = hp.num_baseline_nodes
        # encoder
        self.lstm = nn.LSTM(input_size = self.num_hidden_nodes, hidden_size=self.num_baseline_nodes,
                              num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

        #self.maxpool = nn.MaxPool1d(3, stride=2, ceil_mode = True)
        if hp.ASR_based:
            self.Self_Attention1 = Self_Attention(8)
            self.dense = nn.Linear(self.num_baseline_nodes*16, hp.num_baseline_nodes*4)
            self.output = nn.Linear(self.num_baseline_nodes*4, hp.num_emotion)
        elif hp.attention_type == 'Self_Attention':
            #self.Self_Attention = Self_Attention()
            self.Self_Attention1 = Self_Attention(8)
            self.Self_Attention2 = Self_Attention(8)
            #self.fc = nn.Linear(self.num_hidden_nodes, self.num_baseline_nodes*2)
            #self.dense = nn.Linear(self.num_hidden_nodes*2*hp.head, hp.num_hidden_nodes*2*hp.head)
            #self.output = nn.Linear(self.num_hidden_nodes*2*hp.head,hp.num_emotion)
            self.dense = nn.Linear(self.num_baseline_nodes*16*2, hp.num_baseline_nodes*4*2)
            self.output = nn.Linear(self.num_baseline_nodes*4*2,hp.num_emotion)
            #self.fc_trans = nn.Linear(self.num_baseline_nodes*2*4, hp.num_baseline_nodes*2)
            #self.Self_Attention = nn.MultiheadAttention(self.num_hidden_nodes*2, hp.head, dropout = 0.5)
        elif hp.combined_ASR:
            #self.fc = nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes*2)
            self.fc_text = nn.Linear(self.num_baseline_nodes*2, hp.num_baseline_nodes)
            self.fc_acoustic = nn.Linear(self.num_baseline_nodes*2, hp.num_baseline_nodes)
            self.dense = nn.Linear(self.num_baseline_nodes*2, hp.num_baseline_nodes)
            self.output = nn.Linear(self.num_baseline_nodes,hp.num_emotion)
        else:
            self.fc_text = nn.Linear(self.num_hidden_nodes*2, hp.num_hidden_nodes*2)
            self.fc_acoustic = nn.Linear(self.num_hidden_nodes*2, hp.num_hidden_nodes*2)
            self.dense = nn.Linear(self.num_hidden_nodes*2, hp.num_hidden_nodes*2)
            self.output = nn.Linear(self.num_hidden_nodes*2,hp.num_emotion)
        self.activation = nn.ReLU()
        #self.gs = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes*2)

    def forward(self, sbatch, hbatch, targets):
        #batch_size = hbatch.size(0)
        #num_frames = hbatch.size(1)
        #num_labels = targets.size(1)

        g, _ = self.lstm(sbatch)
        #g = self.fc(sbatch)
        #print(g.size())
        #print(hbatch.size())
        text = g.mean(dim=1)
        #text = g.sum(dim=1)

        if hp.ASR_based and hp.attention_type == 'Self_Attention':
            maxpg = self.Self_Attention1(g)
            maxp = maxpg
        elif hp.attention_type == 'Self_Attention':
            #concat = torch.cat((g,hbatch),1)
            #print(concat.size())
            maxpg = self.Self_Attention1(g)
            maxph = self.Self_Attention2(hbatch)
            #maxph = self.fc_trans(maxph)
            maxp = torch.cat((maxpg,maxph),1)
            #maxp = self.Self_Attention(concat)
            #maxp = self.Self_Attention(concat,concat, concat)

        elif hp.combined_ASR:
            acoustic = hbatch.mean(dim=1)
            #acoustic = hbatch.sum(dim=1)
            maxp = torch.cat((self.fc_text(text),self.fc_acoustic(acoustic)),1)
        else:
            maxp = text

        res = self.activation(self.dense(maxp))

        emotion = self.output(res)
        #print(emotion.size())

        return emotion

    def decode(self, sbatch, hbatch, lengths):
        #batch_size = hbatch.size(0)
        #num_frames = hbatch.size(1)
        #num_labels = targets.size(1)

        #print(sbatch[:,:lengths])
        #print(lengths)
        sbatch = sbatch[:,:lengths]
        g, _ = self.lstm(sbatch)
        #g = self.fc(sbatch)
        #print(g.size())
        #print(hbatch.size())
        text = g.mean(dim=1)
        #text = g.sum(dim=1)

        if hp.ASR_based and hp.attention_type == 'Self_Attention':
            maxpg = self.Self_Attention1(g)
            maxp = maxpg
        elif hp.attention_type == 'Self_Attention':
            #concat = torch.cat((g,hbatch),1)
            #print(concat.size())
            maxpg = self.Self_Attention1(g)
            maxph = self.Self_Attention2(hbatch)
            #maxph = self.fc_trans(maxph)
            maxp = torch.cat((maxpg,maxph),1)
            #maxp = self.Self_Attention(concat)
            #maxp = self.Self_Attention(concat,concat, concat)

        elif hp.combined_ASR:
            acoustic = hbatch.mean(dim=1)
            #acoustic = hbatch.sum(dim=1)
            maxp = torch.cat((self.fc_text(text),self.fc_acoustic(acoustic)),1)
        else:
            maxp = text

        res = self.activation(self.dense(maxp))

        emotion = self.output(res)
        #print(emotion.size())

        emotion = emotion.squeeze()
        if hp.score_func == 'log_softmax':
            emotion = F.log_softmax(emotion, dim=0)
        elif hp.score_func == 'softmax':
            emotion = F.softmax(emotion, dim=0)

        bestidx = emotion.data.argmax().item()

        return bestidx
