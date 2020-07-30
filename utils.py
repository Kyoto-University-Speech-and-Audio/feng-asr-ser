# -*- coding: utf-8 -*-
import copy
import numpy as np
from struct import unpack, pack
import sys
import torch
import torch.nn as nn

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_config():
    for key in hp.__dict__.keys():
        if not '__' in key:
            print('{} = {}'.format(key, eval('hp.'+key)))

def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)

    """
    fh = open(filename, "rb")
    spam = fh.read(12)
    _, _, sampSize, _ = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def frame_stacking(cpudat, stack):
    newlen = int(cpudat.shape[0] / stack)
    cpudat = cpudat[:stack * newlen, :]
    cpudat = np.reshape(cpudat, (newlen, stack, hp.lmfb_dim))
    cpudat = np.reshape(cpudat, (newlen, stack * hp.lmfb_dim)).astype(np.float32)

    return cpudat, newlen

#def down_sampling(x, sample_rate):
#    batch_size = x.size(0).item()
#    newlen = int(x.size(1).item() / sample_rate)
#    feature = x.size(2).item() * newlen
#    x = torch.reshape(x, (-1, newlen, stack))
#    x = torch.reshape(x, (newlen, stack * hp.lmfb_dim)).astype(np.float32)

#    return cpudat, newlen

def onehot(labels, num_output):
    """
    To make onehot vector.
    ex) labels : 3 -> [0, 0, 1, 0, ...]

    Args:
        labels : true label ID
        num_output : the number of entry

    Returns:
        utt_label : one hot vector.
    """
    utt_label = np.zeros((len(labels), num_output), dtype='float32')
    for i in range(len(labels)):
        utt_label[i][labels[i]] = 1.0
    return utt_label

def onehot_dist(labels, num_output):
    """
    To make onehot vector.
    ex) labels : 3 -> [0, 0, 1, 0, ...]

    Args:
        labels : true label ID
        num_output : the number of entry

    Returns:
        utt_label : one hot vector.
    """
    if len(labels) == 2:
        utt_label = np.zeros((1, num_output), dtype='float32')
        utt_label[0][int(labels[0])] = 1.0
    else:
        utt_label = np.zeros((1, num_output), dtype='float32')
        for i in range(len(labels)):
            utt_label[0][int(labels[i])] += 1/3
        #print(utt_label)
        #print(labels)
        #print()

    return utt_label


# sorting lengths order
def sort_pad(batch_size,xs, lengths, ts=None, ts_onehot=None, ts_onehot_LS=None, ts_lengths=None, emotion = None, emo_onehot=None, emo_onehot_LS=None, temp = None, temp_length = None):
    """
    To sort "lengths" order.
    This funtion is needed for "torch.nn.utils.rnn.pack_padded_sequence()"

    Args:
        xs : input feature. (BATCH SIZE, time frames, log mel-scale filter bank)
        ts : grand truth data (BATCH SIZE, label lengths)
        ts_onehot : grand truth data which is an onehot vector (BATCH SIZE, label lengths, #labels)
        ts_onehot_LS : grand truth data which is a vector for label smoothing (BATCH SIZE, label lengths, #labels)
        lengths : the lengths of the input feature (BATCH SIZE)
        ts_lengths : the lengths of grand truth data (BATCH SIZE)

    Returns:
        xs_tensor : "torch FloatTensor" of sorted xs
        ts_results : list of sorted ts
        ts_onehot_tensor : "torch FloatTensor" of sorted ts_onehot
        lengths : sorted lenghts
    """
    def argsort(seq):
        return np.argsort(np.array(seq))[::-1].tolist()
    if hp.text_based or hp.combined:
        arg_lengths = argsort(ts_lengths)
    else:
        arg_lengths = argsort(lengths)
    maxlen = max(lengths)

    if hp.encoder_type != 'Wave':
        input_size = hp.lmfb_dim * hp.frame_stacking if hp.frame_stacking else hp.lmfb_dim
    else:
        input_size = 1

    if (ts is not None) and (ts_lengths is not None) and (ts_onehot_LS is not None) and (emotion is not None) and (emo_onehot_LS is not None) :
        if hp.encoder_type == 'CNN':
            xs_tensor = torch.zeros((batch_size, maxlen, 3, input_size // 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
        else:
            xs_tensor = torch.zeros((batch_size, maxlen, input_size), dtype=torch.float32, device=DEVICE, requires_grad=True)
            xs_temp = torch.zeros((batch_size, max(temp_length), hp.lmfb_dim), dtype=torch.float32, device=DEVICE, requires_grad=True)
        ts_maxlen = max(ts_lengths)
        ts_result = torch.zeros((batch_size, ts_maxlen), dtype=torch.int64, device=DEVICE, requires_grad=False)
        ts_onehot_tensor = torch.zeros((batch_size, ts_maxlen, hp.num_classes), dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        ts_onehot_LS_tensor = torch.zeros((batch_size, ts_maxlen, hp.num_classes), dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        emo_onehot_tensor = torch.zeros((batch_size, hp.num_emotion), dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        emo_onehot_LS_tensor = torch.zeros((batch_size, hp.num_emotion), dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        lengths_tensor = torch.zeros((batch_size), dtype=torch.int64, device=DEVICE)
        ts_lengths_new = copy.deepcopy(ts_lengths)
        #ts_result = []
        emo_result = []
        for i, i_sort in enumerate(arg_lengths):
            xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
            xs_temp.data[i, 0:temp_length[i_sort]] = torch.from_numpy(temp[i_sort])
            ts_onehot_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot[i_sort])
            ts_onehot_LS_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot_LS[i_sort])
            ts_result.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts[i_sort])
            emo_onehot_tensor.data[i] = torch.from_numpy(emo_onehot[i_sort])
            emo_onehot_LS_tensor.data[i] = torch.from_numpy(emo_onehot_LS[i_sort])
            emo_result.extend(list(emotion[i_sort]))
            lengths_tensor.data[i] = lengths[i_sort]
            ts_lengths_new[i] = ts_lengths[i_sort]
        return xs_tensor, lengths_tensor, ts_result, ts_onehot_tensor, ts_onehot_LS_tensor, torch.LongTensor(ts_lengths_new), torch.LongTensor(emo_result), emo_onehot_tensor, emo_onehot_LS_tensor, xs_temp
    elif (ts is not None) and (ts_lengths is not None) and (ts_onehot_LS is not None):
        if hp.encoder_type == 'CNN':
            xs_tensor = torch.zeros((batch_size, maxlen, 3, input_size // 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
        else:
            xs_tensor = torch.zeros((batch_size, maxlen, input_size), dtype=torch.float32, device=DEVICE, requires_grad=True)
        ts_maxlen = max(ts_lengths)
        ts_onehot_tensor = torch.zeros((batch_size, ts_maxlen, hp.num_classes), dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        ts_onehot_LS_tensor = torch.zeros((batch_size, ts_maxlen, hp.num_classes), dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        lengths_tensor = torch.zeros((batch_size), dtype=torch.int64, device=DEVICE)
        ts_lengths_new = copy.deepcopy(ts_lengths)
        ts_result = []
        for i, i_sort in enumerate(arg_lengths):
            xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
            ts_onehot_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot[i_sort])
            ts_onehot_LS_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot_LS[i_sort])
            ts_result.extend(list(ts[i_sort]))
            lengths_tensor.data[i] = lengths[i_sort]
            ts_lengths_new[i] = ts_lengths[i_sort]
        return xs_tensor, lengths_tensor, torch.LongTensor(ts_result), ts_onehot_tensor, ts_onehot_LS_tensor, torch.LongTensor(ts_lengths_new)
    else:
        if hp.encoder_type == 'CNN':
            xs_tensor = torch.zeros((1, maxlen, 3, input_size // 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
        else:
            xs_tensor = torch.zeros((1, maxlen, input_size), dtype=torch.float32, device=DEVICE, requires_grad=True)
            xs_temp = torch.zeros((batch_size, max(temp_length), hp.lmfb_dim), dtype=torch.float32, device=DEVICE, requires_grad=True)
        lengths_tensor = torch.zeros((1), dtype=torch.int64, device=DEVICE)
        for i, i_sort in enumerate(arg_lengths):
            xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
            xs_temp.data[i, 0:temp_length[i_sort]] = torch.from_numpy(temp[i_sort])
            lengths_tensor.data[i] = lengths[i_sort]
        return  xs_tensor, lengths_tensor, xs_temp

def load_model(model_file):
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key] = model_state[key]

        return new_model_state
    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]
        return new_model_state
    else:
        print('ERROR in load model')
        sys.exit(1)

def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

def adjust_learning_rate(optimizer, epoch):
    if epoch > 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
