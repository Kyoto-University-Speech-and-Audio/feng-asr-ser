import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Combined(nn.Module):
    def __init__(self):
        super(Combined, self).__init__()
        self.num_decoder_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.num_baseline_nodes = hp.num_baseline_nodes
        # encoder
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim
        self.emb = nn.Embedding(hp.num_classes,300)
        self.lstm1 = nn.LSTM(input_size = 300, hidden_size=self.num_baseline_nodes,
                              num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size = input_size, hidden_size=self.num_baseline_nodes,
                              num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc_text = nn.Linear(self.num_baseline_nodes*2, hp.num_baseline_nodes*2)
        self.fc_acoustic = nn.Linear(self.num_baseline_nodes*2, hp.num_baseline_nodes*2)

        self.dense = nn.Linear(self.num_baseline_nodes*4, hp.num_baseline_nodes*4)
        #self.maxpool = nn.MaxPool1d(3, stride=2, ceil_mode = True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(self.num_baseline_nodes*4,hp.num_emotion)

    def forward(self, x, lengths, ts, ts_lengths, seq1, seq2):
        ts = self.emb(ts)
        total_length_ts = ts.size(1)
        ts = nn.utils.rnn.pack_padded_sequence(ts, ts_lengths, batch_first=True)
        h_ts, _ = self.lstm1(ts)
        hbatch_ts, ts_lengths = nn.utils.rnn.pad_packed_sequence(h_ts, batch_first=True, total_length=total_length_ts)
        #print(hbatch.size())
        text = torch.zeros((hp.batch_size, hp.num_baseline_nodes*2), dtype=torch.float32, device=DEVICE)
        for i, i_sort in enumerate(ts_lengths):
            text[i,:]=hbatch_ts[i].mean(dim=0)

        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        h, _ = self.lstm2(x)
        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        #print(hbatch.size())
        acoustic = torch.zeros((hp.batch_size, hp.num_baseline_nodes*2), dtype=torch.float32, device=DEVICE)
        for i, i_sort in enumerate(lengths):
            acoustic[i,:]=hbatch[i].mean(dim=0)
        #maxp = self.maxpool(hbatch)

        for i in range(hp.batch_size):
            for j in range(hp.batch_size):
                if seq2[j] == i:
                    break
            temp = text[i]
            text[i] = text[j]
            text[j] = temp

        for i in range(hp.batch_size):
            for j in range(hp.batch_size):
                if seq1[j] == i:
                    break
            temp = acoustic[i]
            acoustic[i] = acoustic[j]
            acoustic[j] = temp

        maxp = torch.cat((self.fc_text(text),self.fc_acoustic(acoustic)),1)

        res = self.activation(self.dense(maxp))
        #print(res.size())
        emotion = self.output(res)
        #print(emotion.size())

        return emotion

    def decode(self, x, lengths, ts, ts_lengths):
        with torch.no_grad():
            ts = self.emb(ts)
            total_length_ts = ts.size(1)
            ts = nn.utils.rnn.pack_padded_sequence(ts, ts_lengths, batch_first=True)
            h_ts, _ = self.lstm1(ts)
            hbatch_ts, ts_lengths = nn.utils.rnn.pad_packed_sequence(h_ts, batch_first=True, total_length=total_length_ts)
            #print(hbatch.size())
            text = torch.zeros((1, hp.num_baseline_nodes*2), dtype=torch.float32, device=DEVICE)
            for i, i_sort in enumerate(ts_lengths):
                text[i,:]=hbatch_ts[i].mean(dim=0)

            total_length = x.size(1)
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            h, _ = self.lstm2(x)
            hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
            #print(hbatch.size())
            acoustic = torch.zeros((1, hp.num_baseline_nodes*2), dtype=torch.float32, device=DEVICE)
            for i, i_sort in enumerate(lengths):
                acoustic[i,:]=hbatch[i].mean(dim=0)
            #maxp = self.maxpool(hbatch)

            maxp = torch.cat((self.fc_text(text),self.fc_acoustic(acoustic)),1)

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
