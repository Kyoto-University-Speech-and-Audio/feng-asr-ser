# -*- coding: utf-8 -*-

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer, \
                batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x, lengths):
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        self.bi_lstm.flatten_parameters()
        h, _ = self.bi_lstm(x)
        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        return hbatch

class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        # encoder_cnn
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3),
                        stride = (2, 2), padding = (1, 0))
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3),
                        stride = (2, 2), padding = (1, 0))
        self.conv2_bn = nn.BatchNorm2d(32)
        # encoder
        self.bi_lstm = nn.LSTM(input_size=640, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer,
                         batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        conv_out = self.conv1(x.permute(0, 2, 3, 1))
        batched = self.conv1_bn(conv_out)
        activated = F.relu(batched)
        conv_out = self.conv2(activated)
        batched = self.conv2_bn(conv_out)
        activated = F.relu(batched)

        cnnout = activated.permute(0, 3, 1, 2).reshape(batch_size, activated.size(3), -1)

        newlengths = []
        for xlen in lengths.cpu().numpy():
            q1, mod1 = divmod(xlen, 2)
            if mod1 == 0:
                xlen1 = xlen // 2 - 1
                q2, mod2 = divmod(xlen1, 2)
                if mod2 == 0:
                    xlen2 = xlen1 // 2 - 1
                else:
                    xlen2 = (xlen1 - 1) // 2
            else:
                xlen1 = (xlen - 1) // 2
                q2, mod2 = divmod(xlen1, 2)
                if mod2 == 0:
                    xlen2 = xlen1 // 2 - 1
                else:
                    xlen2 = (xlen1 - 1) // 2
            newlengths.append(xlen2)

        cnnout_packed = nn.utils.rnn.pack_padded_sequence(cnnout, newlengths, batch_first=True)

        h, _ = self.bi_lstm(cnnout_packed)

        hbatch, newlengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return hbatch

class WaveEncoder(nn.Module):
    def __init__(self):
        super(WaveEncoder, self).__init__()
        ## frond-end part
        self.epsilon = 1e-8
        # Like preemphasis filter
        self.preemp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
        # init
        tmp = torch.zeros((1,1,2)).to(DEVICE)
        tmp.data[:,:,0] = -0.97
        tmp.data[:,:,1] = 1
        self.preemp.weight.data = torch.tensor(tmp)

        # if 16kHz
        self.comp = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=400, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.comp.weight.data)

        # B x 400 (0.01s = 10ms)
        tmp = np.zeros((40, 1, 400))
        tmp[:, :] = scipy.hanning(400 + 1)[:-1]
        tmp = tmp * tmp

        K = torch.tensor(tmp, dtype=torch.float).to(DEVICE)

        self.lowpass_weight = K

        self.instancenorm = nn.InstanceNorm1d(40)

        # encoder part
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim

        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer, \
                batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x_waveform, lengths_waveform):

        x_preemp = self.preemp(x_waveform.permute(0,2,1))
        x_comp = self.comp(x_preemp)

        x_even = x_comp[:, 0::2, :]
        x_odd = x_comp[:, 1::2, :]
        x_abs = torch.sqrt(x_even * x_even + x_odd * x_odd + self.epsilon)

        x_lowpass = F.conv1d(x_abs, self.lowpass_weight, stride=160, groups=40)

        x_log = torch.log(1.0 + torch.abs(x_lowpass))

        x_norm = self.instancenorm(x_log).permute(0,2,1)

        x_lengths = lengths_waveform - 1
        x_lengths = (x_lengths - (400 - 1)) // 1
        x_lengths = (x_lengths - (400 - 160)) // 160

        seqlen = x_norm.shape[1]

        if hp.frame_stacking:
            if seqlen % 3 == 0:
               x_norm = torch.cat((x_norm[:, 0::3], x_norm[:, 1::3], x_norm[:, 2::3, :]), dim=2)
            elif seqlen % 3 == 1:
                x_norm = torch.cat((x_norm[:, 0:-1:3,:], x_norm[:, 1::3, :], x_norm[:, 2::3, :]), dim=2)
            elif seqlen % 3 == 2:
                x_norm = torch.cat((x_norm[:, 0:-2:3,:], x_norm[:, 1:-1:3, :], x_norm[:, 2::3, :]), dim=2)

            x_lengths /= 3

        x = nn.utils.rnn.pack_padded_sequence(x_norm, x_lengths.tolist(), batch_first=True)

        h, (_, _) = self.bi_lstm(x)

        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        return hbatch
