# -*- coding: utf-8 -*-
import copy
import numpy as np
from operator import itemgetter, attrgetter
import torch
import torch.nn.functional as F
import torch.nn as nn

from Models.attention import Attention
import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.num_decoder_hidden_nodes = hp.num_hidden_nodes
        self.num_classes = hp.num_classes
        self.att = Attention()
        # decoder
        self.L_sy = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False)
        self.L_gy = nn.Linear(self.num_decoder_hidden_nodes * 2, self.num_decoder_hidden_nodes)
        self.L_yy = nn.Linear(self.num_decoder_hidden_nodes, self.num_classes)

        self.L_ys = nn.Linear(self.num_classes, self.num_decoder_hidden_nodes * 4 , bias=False)
        self.L_ss = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes * 4, bias=False)
        self.L_gs = nn.Linear(self.num_decoder_hidden_nodes * 2, self.num_decoder_hidden_nodes * 4)

    def forward(self, hbatch, lengths, targets):
        batch_size = hbatch.size(0)
        num_frames = hbatch.size(1)
        num_labels = targets.size(1)

        e_mask = torch.ones((batch_size, num_frames, 1), device=DEVICE, requires_grad=False)
        s = torch.zeros((batch_size, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False)
        c = torch.zeros((batch_size, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False)

        youtput = torch.zeros((batch_size, num_labels, self.num_classes), device=DEVICE, requires_grad=False)
        alpha = torch.zeros((batch_size, 1,  num_frames), device=DEVICE, requires_grad=False)
        hidden_state = torch.zeros((batch_size, num_labels, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False)

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask.data[i, tmp:] = 0.0

        for step in range(num_labels):
            g, alpha = self.att(s, hbatch, alpha, e_mask)
            # generate
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            # recurrency calcuate
            rec_input = self.L_ys(targets[:,step,:]) + self.L_ss(s) + self.L_gs(g)
            s, c = self._func_lstm(rec_input, c)

            #hidden_state[:,step,:] = s
            hidden_state[:,step,:] = torch.tanh(self.L_gy(g) + self.L_sy(s))
            youtput[:,step] = y
        if hp.ASR:
            return youtput
        else:
            return youtput, hidden_state

    def decode(self, hbatch, lengths):
        batch_size = hbatch.size(0)
        num_frames = hbatch.size(1)
        e_mask = torch.ones((batch_size, num_frames, 1), device=DEVICE, requires_grad=False)

        token_beam_sel = [([], 0.0, (torch.zeros((batch_size, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, 1, num_frames), device=DEVICE, requires_grad=False)),
                        torch.zeros((batch_size, hp.max_decoder_seq_len, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, hp.max_decoder_seq_len, self.num_decoder_hidden_nodes), device=DEVICE, requires_grad=False))]

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask[i, tmp:] = 0.0

        step = -1
        for _ in range(hp.max_decoder_seq_len):
            token_beam_all = []
            step += 1

            for current_token in token_beam_sel:
                cand_seq, cand_seq_score, (c, s, alpha), cand_seq_s, cand_seq_hidden = current_token

                g, alpha = self.att(s, hbatch, alpha, e_mask)

                # generate
                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

                if hp.score_func == 'log_softmax':
                    y = F.log_softmax(y, dim=1)
                elif hp.score_func == 'softmax':
                    y = F.softmax(y, dim=1)

                tmpy = y.clone()
                for _ in range(hp.beam_width):
                    bestidx = tmpy.data.argmax(1).item()
                    #print(bestidx)

                    tmpseq = cand_seq.copy()
                    tmpseq.append(bestidx)

                    tmpscore = cand_seq_score + tmpy.data[0][bestidx]
                    tmpy.data[0][bestidx] = -10000000000.0
                    target_for_t_estimated = torch.zeros((1, hp.num_classes), device=DEVICE, requires_grad=False)

                    target_for_t_estimated.data[0][bestidx] = 1.0
                    rec_input = self.L_ys(target_for_t_estimated) + self.L_ss(s) + self.L_gs(g)
                    tmps, tmpc = self._func_lstm(rec_input, c)
                    tmp_seq_s = copy.deepcopy(cand_seq_s)
                    tmp_seq_hidden = copy.deepcopy(cand_seq_hidden)
                    tmp_seq_s[:,step] = s
                    tmp_seq_hidden[:,step] = torch.tanh(self.L_gy(g) + self.L_sy(s))

                    token_beam_all.append((tmpseq, tmpscore, (tmpc, tmps, alpha), tmp_seq_s, tmp_seq_hidden))
            sorted_token_beam_all = sorted(token_beam_all, key=itemgetter(1), reverse=True)
            token_beam_sel = sorted_token_beam_all[:hp.beam_width]
            #print(token_beam_sel[0][0])
            results = []
            if token_beam_sel[0][0][-1] == hp.eos_id:
                for character in token_beam_sel[0][0]:
                    results.append(character)
                break
        if hp.ASR:
            return results
        else:
            #print(token_beam_sel[0][3][0][4])
            #print(token_beam_sel[0][4])
            return results, token_beam_sel[0][4]

    @staticmethod
    def _func_lstm(x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        half = 0.5
        ingate = torch.tanh(ingate * half) * half + half
        forgetgate = torch.tanh(forgetgate * half) * half + half
        cellgate = torch.tanh(cellgate)
        outgate = torch.tanh(outgate * half) * half + half
        c_next = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c_next)
        return h, c_next
