import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import hparams as hp
from Models.Self_Attention import Self_Attention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class text_based(nn.Module):
    def __init__(self):
        super(text_based, self).__init__()
        self.num_decoder_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.num_baseline_nodes = hp.num_baseline_nodes
        # encoder
        self.emb = nn.Embedding(hp.num_classes,300)
        self.lstm = nn.LSTM(input_size = 300, hidden_size=self.num_baseline_nodes,
                              num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(self.num_baseline_nodes*8*2, hp.num_baseline_nodes*2*2)
        #self.maxpool = nn.MaxPool1d(3, stride=2, ceil_mode = True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(self.num_baseline_nodes*2*2,hp.num_emotion)
        if hp.attention_type == 'Self_Attention':
            self.Self_Attention1 = Self_Attention(8)

    def forward(self, x, lengths):
        x = self.emb(x)
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        h, _ = self.lstm(x)
        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
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
            x = self.emb(x)
            total_length = x.size(1)
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            h, _ = self.lstm(x)
            hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
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
