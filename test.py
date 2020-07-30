# -*- coding: utf-8 -*-
import argparse
import copy
import itertools
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp
from Models.AttModel import AttModel
from Models.Baseline import Baseline
from Models.text_based import text_based
from Models.Combined import Combined
from utils import frame_stacking, onehot, load_dat, log_config, sort_pad, load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_loop(model, test_set):
    batch_size = 1
    #mean = np.load("/n/work1/feng/src/htk/mean.npy")
    #var = np.load("/n/work1/feng/src/htk/var.npy")
    acc = 0
    neutral = 0
    positive = 0
    negative = 0
    ang = 0
    ang_total = 0
    neutral_total = 0
    positive_total = 0
    negative_total = 0
    total = 0
    confusion_matrix = np.zeros((4,4))
    for i in range(len(test_set)):
        # input lmfb (B x T x (F x frame_stacking))
        xs = []
        # target symbols
        ts = []
        # onehot vector of target symbols (B x L x NUM_CLASSES)
        ts_onehot = []
        # vector of target symbols for label smoothing (B x L x NUM_CLASSES)
        ts_onehot_LS = []
        # input lengths
        emo = []
        emo_onehot = []
        emo_onehot_LS = []

        lengths = []
        ts_lengths = []
        # input lmfb (B x T x (F x frame_stacking))
        xs1 = []
        # target symbols
        ts1 = []
        # onehot vector of target symbols (B x L x NUM_CLASSES)
        ts_onehot1 = []
        # vector of target symbols for label smoothing (B x L x NUM_CLASSES)
        ts_onehot_LS1 = []
        # input lengths
        emo1 = []
        emo_onehot1 = []
        emo_onehot_LS1 = []

        lengths1 = []
        ts_lengths1 = []
        temp = []
        temp_length = []

        for j in range(batch_size):
            s = test_set[i*batch_size+j].strip()
            #if hp.ASR:
            #    x_file = s.strip()
            #else:
            #x_file, laborg = s.split(' ', 1)
            if hp.ASR:
                x_file, laborg = s.split(' ', 1)
            elif hp.dist:
                x_file, laborg, labemo, labdist = s.split('\t')
                laborg = laborg.strip()
                labemo = labemo.strip()
                labdist = labdist.strip()
            else:
                x_file, laborg, labemo = s.split('\t')
                laborg = laborg.strip()
                labemo = labemo.strip()
                #if len(laborg) == 0:
                #    laborg = "2 0 1"
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
                #cpudat = (cpudat-mean)/var
            elif '.npy'in x_file:
                cpudat = np.load(x_file)
                #cpudat = (cpudat-mean)/var
            elif '.wav' in x_file:
                with wave.open(x_file) as wf:
                    dat = wf.readframes(wf.getnframes())
                    y = fromstring(dat, dtype=int16)[:, np.newaxis]
                    y_float = y.astype(np.float32)
                    cpudat = (y_float - np.mean(y_float)) / np.std(y_float)

            tmp = copy.deepcopy(cpudat)
            print(x_file, end='\t')
            if hp.frame_stacking > 1 and hp.encoder_type != 'Wave':
                cpudat, newlen = frame_stacking(cpudat, hp.frame_stacking)

            newlen = cpudat.shape[0]
            if hp.encoder_type == 'CNN':
                cpudat_split = np.split(cpudat, 3, axis = 1)
                cpudat = np.hstack((cpudat_split[0].reshape(newlen, 1, 80),
                            cpudat_split[1].reshape(newlen, 1, 80), cpudat_split[2].reshape(newlen, 1, 80)))
            newlen = cpudat.shape[0]
            lengths.append(newlen)
            xs.append(cpudat)
            temp.append(tmp)
            temp_length.append(tmp.shape[0])

            if hp.ASR == False:
                cpuemo = np.array([int(x) for x in labemo],dtype=np.int32)
                emotion_onehot = onehot(cpuemo, hp.num_emotion)
                emo_onehot.append(emotion_onehot)
                emo_onehot_LS.append(0.9 * emotion_onehot + 0.1 * 1.0 / hp.num_emotion)
                emo.append(cpuemo)
                cpulab = np.array([int(i) for i in laborg.split(' ')], dtype=np.int32)
                cpulab_onehot = onehot(cpulab, hp.num_classes)
                ts.append(cpulab)
                ts_lengths.append(len(cpulab))
                ts_onehot.append(cpulab_onehot)
                ts_onehot_LS.append(0.9 * cpulab_onehot + 0.1 * 1.0 / hp.num_classes)

        if hp.baseline_type != 'lim_BLSTM':
            temp, temp_length = xs, lengths

        if hp.ASR:
            xs, lengths, temp = sort_pad(1, xs, lengths, temp = temp, temp_length = temp_length)
        else:
            xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp = sort_pad(1, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp, temp_length)

        if hp.baseline_type == 'CNN_BLSTM' or hp.baseline_type == 'lim_BLSTM':
            onehot_length = temp.size(2)
            xs_new = torch.zeros((1, 750, onehot_length))
            for i in range(1):
                feature_length = temp.size(1)
                if feature_length > 750:
                    xs_new.data[:,:750,:] = temp.data[:,:750,:]
                else:
                    xs_new.data[:,:feature_length,:] = temp.data[:,:feature_length,:]
        if hp.ASR:
            results = model.decode(xs, lengths, [])
            for character in results:
                print(character, end=' ')
            if results == []:
                print("2 1", end=' ')
            print("\t", end='')
            print(labemo)
            #print()
            sys.stdout.flush()
        elif hp.baseline or hp.combined or hp.text_based:
            if hp.baseline:
                emotion = model.decode(xs_new.to(DEVICE), [])
            elif hp.text_based:
                emotion = model.decode(ts.to(DEVICE), ts_lengths.to(DEVICE))
            elif hp.combined:
                #emotion_in_Variable = model(xs, lengths, ts.to(DEVICE), ts_lengths.to(DEVICE), seq1, seq2)
                emotion = model.decode(xs_new.to(DEVICE), ts.to(DEVICE), ts_lengths.to(DEVICE))
                #emotion = model.decode(xs, lengths, ts1.to(DEVICE), ts_lengths1.to(DEVICE))

            print(int(labemo.strip()), end = '\t')
            print((emotion), end = '\t')
            print()
        else:
            results, emotion = model.decode(xs, lengths, xs_new.to(DEVICE))
            for character in results:
                print(character, end=' ')
            if results == []:
                print("2 1", end=' ')
            print('\t',end='')
            print(int(labemo.strip()), end = '\t')

            print(emotion, end = '\t')
            print()
            sys.stdout.flush()
        #if int(labemo.strip())==2:
        #    neutral_total += 1
        #elif int(labemo.strip())==0:
        #    positive_total += 1
        #elif int(labemo.strip())==1:
        #    negative_total += 1
        #elif int(labemo.strip())==3:
        #    ang_total += 1

        #if emotion == int(labemo.strip()):
        #    acc += 1
        #    if emotion == 2:
        #        neutral += 1
        #    elif emotion == 0:
        #        positive += 1
        #    elif emotion == 1:
        #        negative += 1
        #    elif emotion == 3:
        #        ang += 1
        #total += 1
        #confusion_matrix[int(labemo.strip()), emotion] += 1
    #if hp.ASR == False and hp.combined_ASR == False and hp.ASR_based == False:
    #if hp.ASR == False:
    #    res = acc/total
    #    print("Accuracy of emotion ", res, acc, total)
    #    res = neutral/neutral_total
    #    print("neu:"+str(res), neutral, neutral_total)
    #    res = positive/positive_total
    #    print("hap:"+str(res), positive, positive_total)
    #    res = negative/negative_total
    #    print("sad:"+str(res), negative, negative_total)
    #    print("ang:"+str(ang/ang_total), ang, ang_total)

if __name__ == "__main__":
    if hp.baseline:
        model = Baseline().to(DEVICE)
    elif hp.text_based:
        model = text_based().to(DEVICE)
    elif hp.combined:
        model = Combined().to(DEVICE)
    else:
        model = AttModel().to(DEVICE)

    model.eval()
    parser = argparse.ArgumentParser()
    #parser.add_argument('--load_model', default=hp.load_checkpoints_path+'/network.epoch{}'.format(hp.load_checkpoints_epoch))
    parser.add_argument('--load_name')
    parser.add_argument('--test')
    args = parser.parse_args()
    load_name = args.load_name
    test_script = args.test

    test_set = []
    with open(test_script) as f:
        for line in f:
            test_set.append(line)

    #assert hp.load_checkpoints, 'Please specify the checkpoints'
    # consider load methods

    model.load_state_dict(load_model(load_name))
    test_loop(model, test_set)
