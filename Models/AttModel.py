# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from Models.decoder import Decoder
from Models.encoder import Encoder, CNN_Encoder, WaveEncoder
from Models.wordlevelencoder import Wordlevelencoder
from Models.text_based import text_based
from Models.Acencoder import Acencoder

import hparams as hp

class AttModel(nn.Module):
    def __init__(self):
        super(AttModel, self).__init__()
        if hp.encoder_type == 'CNN':
            self.encoder = CNN_Encoder()
        elif hp.encoder_type == 'Wave':
            self.encoder = WaveEncoder()
        else:
            self.encoder = Encoder()
        self.decoder = Decoder()
        if hp.combined_ASR:
            self.wordlevel = Wordlevelencoder()
            self.acencoder = Acencoder()
        if hp.ASR_based:
            self.wordlevel = Wordlevelencoder()

    def forward(self, x, lengths, targets, gtemotion, x_new):
        hbatch = self.encoder(x, lengths)
        if hp.ASR:
            youtput = self.decoder(hbatch, lengths, targets)
            return youtput
        elif hp.combined_ASR:
            youtput, hidden_state = self.decoder(hbatch,lengths,targets)
            if hp.baseline_type == 'CNN_BLSTM' or hp.baseline_type == 'lim_BLSTM':
                sbatch = self.acencoder(x_new, lengths)
            else:
                sbatch = self.acencoder(x, lengths)
            #print(sbatch)
            emotion = self.wordlevel(hidden_state, sbatch, gtemotion)
            #emotion = self.wordlevel(hidden_state, [], gtemotion)
        elif hp.ASR_based:
            youtput, hidden_state = self.decoder(hbatch,lengths,targets)

            emotion = self.wordlevel(hidden_state, [], gtemotion)

        return youtput, emotion

    def decode(self, x, lengths, xs_new):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            if hp.ASR:
                results = self.decoder.decode(hbatch, lengths)
                return results
            elif hp.combined_ASR:
                results, hidden_state = self.decoder.decode(hbatch, lengths)
                length_results = len(results)
                if length_results == 0:
                    return results, -1
                if hp.baseline_type == 'CNN_BLSTM' or hp.baseline_type == 'lim_BLSTM':
                    sbatch = self.acencoder(xs_new, [])
                else:
                    sbatch = self.acencoder(x, lengths)
                emotion = self.wordlevel.decode(hidden_state, sbatch, length_results)
            else:
                results, hidden_state = self.decoder.decode(hbatch,lengths)
                length_results = len(results)
                if length_results == 0:
                    return results, -1

                emotion = self.wordlevel.decode(hidden_state, [], length_results)

            return results, emotion
