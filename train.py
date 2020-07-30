# -*- coding: utf-8 -*-
import copy
import glob
import itertools
import numpy as np
import os
from scipy import fromstring, int16
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import wave
import random
import argparse

import hparams as hp
from Models.AttModel import AttModel
from Models.Baseline import Baseline
from Models.text_based import text_based
from Models.Combined import Combined
from utils import frame_stacking, onehot, load_dat, log_config, sort_pad, load_model, init_weight, adjust_learning_rate, onehot_dist
from Loss.label_smoothing import label_smoothing_loss


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimizer, train_set, scheduler=None):
    num_mb = len(train_set) // hp.batch_size

    if scheduler:
        scheduler.step(epoch)

    for i in range(num_mb):
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
        temp = []
        temp_length = []
        for j in range(hp.batch_size):
            s = train_set[i*hp.batch_size+j].strip()
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
                #mean = np.load("/n/work1/feng/src/htk/mean.npy")
                #var = np.load("/n/work1/feng/src/htk/var.npy")
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
                #cpudat = (cpudat-mean)/var
                #print(mean)
            elif '.npy'in x_file:
                #mean = np.load("/n/work1/feng/data/swb/mean.npy")
                #var = np.load("/n/work1/feng/data/swb/var.npy")
                cpudat = np.load(x_file)
                #cpudat = (cpudat-mean)/var
            elif '.wav' in x_file:
                with wave.open(x_file) as wf:
                    dat = wf.readframes(wf.getnframes())
                    y = fromstring(dat, dtype=int16)[:, np.newaxis]
                    y_float = y.astype(np.float32)
                    cpudat = (y_float - np.mean(y_float)) / np.std(y_float)

            tmp = copy.deepcopy(cpudat)
            print("{} {}".format(x_file, cpudat.shape[0]))
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

            cpulab = np.array([int(i) for i in laborg.split(' ')], dtype=np.int32)
            #print(cpulab)

            cpulab_onehot = onehot(cpulab, hp.num_classes)
            ts.append(cpulab)
            ts_lengths.append(len(cpulab))
            ts_onehot.append(cpulab_onehot)
            ts_onehot_LS.append(0.9 * cpulab_onehot + 0.1 * 1.0 / hp.num_classes)
            if hp.dist and hp.ASR == False:
                cpuemo = np.array([int(x) for x in labemo],dtype=np.int32)
                emotion_onehot = onehot_dist(labdist, hp.num_emotion)
                emo_onehot.append(emotion_onehot)
                emo_onehot_LS.append(0.9 * emotion_onehot + 0.1 * 1.0 / hp.num_emotion)
                emo.append(cpuemo)
            elif hp.ASR == False:
                cpuemo = np.array([int(x) for x in labemo],dtype=np.int32)
                emotion_onehot = onehot(cpuemo, hp.num_emotion)
                emo_onehot.append(emotion_onehot)
                emo_onehot_LS.append(0.9 * emotion_onehot + 0.1 * 1.0 / hp.num_emotion)
                emo.append(cpuemo)

        if hp.baseline_type != 'lim_BLSTM':
            temp, temp_length = xs, lengths

        if hp.ASR:
            xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths = sort_pad(hp.batch_size, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths)

            youtput_in_Variable = model(xs, lengths, ts_onehot, [], [])

            loss = 0.0
            if hp.decoder_type == 'Attention':
                for k in range(hp.batch_size):
                    num_labels = ts_lengths[k]
                    loss += label_smoothing_loss(youtput_in_Variable[k][:num_labels], ts_onehot_LS[k][:num_labels],1) / num_labels
            print('loss = {}'.format(loss.item()))
        elif hp.baseline:
            xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp = sort_pad(hp.batch_size, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp, temp_length)

            if hp.baseline_type == 'CNN_BLSTM' or hp.baseline_type == 'lim_BLSTM':
                onehot_length = temp.size(2)
                xs_new = torch.zeros((hp.batch_size, 750, onehot_length))
                for i in range(hp.batch_size):
                    feature_length = temp.size(1)
                    if feature_length > 750:
                        xs_new.data[:,:750,:] = temp.data[:,:750,:]
                    else:
                        xs_new.data[:,:feature_length,:] = temp.data[:,:feature_length,:]
                emotion_in_Variable = model(xs_new.to(DEVICE), [])
            else:
                #youtput_in_Variable, emotion_in_Variable = model(xs, lengths, ts_onehot, emo_onehot, [])
                emotion_in_Variable = model(xs, lengths)

            loss = 0.0
            if hp.decoder_type == 'Attention':
                #print(emo)
                #print(emotion_in_Variable[:,:hp.num_emotion])
                loss += F.cross_entropy(emotion_in_Variable[:,:hp.num_emotion], emo.to(DEVICE))
                #for k in range(hp.batch_size):
                    #num_labels = ts_lengths[k]
                    #loss += label_smoothing_loss(youtput_in_Variable[k][:num_labels], ts_onehot_LS[k][:num_labels],1) / num_labels
                    #print(emotion_in_Variable[k][:hp.num_emotion])
                    #loss += F.cross_entropy(emotion_in_Variable[k][:hp.num_emotion], emo)
            print('loss = {}'.format(loss.item()))
        elif hp.text_based:
            xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp = sort_pad(hp.batch_size, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp, temp_length)

            emotion_in_Variable = model(ts.to(DEVICE), ts_lengths.to(DEVICE))

            loss = 0.0
            if hp.decoder_type == 'Attention':
                #print(emo)
                #print(emotion_in_Variable[:,:hp.num_emotion])
                loss += F.cross_entropy(emotion_in_Variable[:,:hp.num_emotion], emo.to(DEVICE))
                #for k in range(hp.batch_size):
                    #num_labels = ts_lengths[k]
                    #loss += label_smoothing_loss(youtput_in_Variable[k][:num_labels], ts_onehot_LS[k][:num_labels],1) / num_labels
                    #print(emotion_in_Variable[k][:hp.num_emotion])
                    #loss += F.cross_entropy(emotion_in_Variable[k][:hp.num_emotion], emo)
            print('loss = {}'.format(loss.item()))
        elif hp.combined:
            #seq1 = []
            #seq2 = []
            #seq1, seq2, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, \
            #xs1, lengths1, ts1, ts_onehot1, ts_onehot_LS1, ts_lengths1, emo1, emo_onehot1, emo_onehot_LS1 \
            #= sort_pad(hp.batch_size, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS)
            xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp = sort_pad(hp.batch_size, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp, temp_length)

            if hp.baseline_type == 'CNN_BLSTM' or hp.baseline_type == 'lim_BLSTM':
                onehot_length = temp.size(2)
                xs_new = torch.zeros((hp.batch_size, 750, onehot_length))
                for i in range(hp.batch_size):
                    feature_length = temp.size(1)
                    if feature_length > 750:
                        xs_new.data[:,:750,:] = temp.data[:,:750,:]
                    else:
                        xs_new.data[:,:feature_length,:] = temp.data[:,:feature_length,:]
                emotion_in_Variable = model(xs_new.to(DEVICE), [], ts.to(DEVICE), ts_lengths.to(DEVICE))
            else:
                emotion_in_Variable = model(xs.to(DEVICE), lengths, ts.to(DEVICE), ts_lengths.to(DEVICE))


            loss = 0.0
            if hp.decoder_type == 'Attention':
                #print(emo)
                #print(emotion_in_Variable[:,:hp.num_emotion])
                #for i in range(hp.batch_size):
                #    for j in range(hp.batch_size):
                #        if seq1[j] == i:
                #            break
                #    temp = emo[i]
                #    emo[i] = emo[j]
                #    emo[j] = temp
                loss += F.cross_entropy(emotion_in_Variable[:,:hp.num_emotion], emo.to(DEVICE))
                #for k in range(hp.batch_size):
                    #num_labels = ts_lengths[k]
                    #loss += label_smoothing_loss(youtput_in_Variable[k][:num_labels], ts_onehot_LS[k][:num_labels],1) / num_labels
                    #print(emotion_in_Variable[k][:hp.num_emotion])
                    #loss += F.cross_entropy(emotion_in_Variable[k][:hp.num_emotion], emo)
            print('loss = {}'.format(loss.item()))
        elif hp.combined_ASR or hp.ASR_based:
            xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp = sort_pad(hp.batch_size, xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths, emo, emo_onehot, emo_onehot_LS, temp, temp_length)

            if hp.baseline_type == 'CNN_BLSTM' or hp.baseline_type == 'lim_BLSTM':
                onehot_length = temp.size(2)
                xs_new = torch.zeros((hp.batch_size, 750, onehot_length))
                for i in range(hp.batch_size):
                    feature_length = temp.size(1)
                    if feature_length > 750:
                        xs_new.data[:,:750,:] = temp.data[:,:750,:]
                    else:
                        xs_new.data[:,:feature_length,:] = temp.data[:,:feature_length,:]
                youtput_in_Variable, emotion_in_Variable = model(xs, lengths, ts_onehot, emo_onehot, xs_new.to(DEVICE))
            else:
                youtput_in_Variable, emotion_in_Variable = model(xs, lengths, ts_onehot, emo_onehot, [])

            loss = 0.0
            if hp.decoder_type == 'Attention':
                #print(emo)
                #print(emotion_in_Variable[:,:hp.num_emotion])
                loss += F.cross_entropy(emotion_in_Variable[:,:hp.num_emotion], emo.to(DEVICE))*0.8
                print(loss)
                for k in range(hp.batch_size):
                    num_labels = ts_lengths[k]
                    loss += label_smoothing_loss(youtput_in_Variable[k][:num_labels], ts_onehot_LS[k][:num_labels],1) / num_labels * 0.2
                    #print(emotion_in_Variable[k][:hp.num_emotion])
                    #loss += F.cross_entropy(emotion_in_Variable[k][:hp.num_emotion], emo)
            print('loss = {}'.format(loss.item()))


        sys.stdout.flush()
        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 1.0
        torch.nn.utils.clip_grad_value_(model.parameters(), clip)
        # optimizer update
        optimizer.step()
        loss.detach()
        torch.cuda.empty_cache()

def train_epoch(model, optimizer, train_set, save_dir, scheduler=None, start_epoch=0):
    if hp.pretrained == True:
        start_epoch = 0
    for epoch in range(start_epoch, hp.max_epoch+1):
        random.shuffle(train_set)
        train_loop(model, optimizer, train_set, scheduler)
        torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
        torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        adjust_learning_rate(optimizer, epoch+1)
        print("EPOCH {} end".format(epoch+1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--save_dir')
    args = parser.parse_args()
    train_script = args.train
    save_dir = args.save_dir

    log_config()
    if hp.baseline:
        model = Baseline()
    elif hp.text_based:
        model = text_based()
    elif hp.combined:
        model = Combined()
    elif hp.decoder_type == 'Attention':
        model = AttModel()

    model.apply(init_weight)

    if torch.cuda.device_count() > 1:
        # multi-gpu configuration
        ngpu = torch.cuda.device_count()
        device_ids = list(range(ngpu))
        model = torch.nn.DataParallel(model, device_ids)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    os.makedirs(save_dir, exist_ok=True)

    load_epoch = 0
    if hp.load_checkpoints and hp.baseline == False and hp.text_based == False and hp.combined == False:
        if hp.load_checkpoints_epoch is None:
            path_list = glob.glob(os.path.join(hp.load_checkpoints_path, 'network.epoch*'))
            for path in path_list:
                epoch = int(path.split('.')[-1].replace('epoch', ''))
                if epoch > load_epoch:
                    load_epoch = epoch
        else:
            load_epoch = hp.load_checkpoints_epoch
        print("{} epoch {} load".format(hp.load_checkpoints_path, load_epoch))
        if hp.pretrained:
            new_pretrained = load_model(os.path.join(hp.load_checkpoints_path, 'network.epoch{}'.format(load_epoch)))
            if new_pretrained == {}:
                exit()
            model_dict = model.state_dict()
            model_dict.update(new_pretrained)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(load_model(os.path.join(hp.load_checkpoints_path, 'network.epoch{}'.format(load_epoch))))
            optimizer.load_state_dict(torch.load(os.path.join(hp.load_checkpoints_path, 'network.optimizer.epoch{}'.format(load_epoch))))
        #model.load_state_dict(load_model(path))
        #optimizer.load_state_dict(torch.load(os.path.join(hp.load_checkpoints_path, 'network.optimizer.epoch{}'.format(load_epoch))))

    train_set = []
    with open(train_script) as f:
        for line in f:
            train_set.append(line)


    train_epoch(model, optimizer, train_set, save_dir, start_epoch=load_epoch)
