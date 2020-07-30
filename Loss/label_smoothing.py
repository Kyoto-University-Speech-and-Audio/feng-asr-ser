# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp

# class LabelSmoothingLoss(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingLoss, self).__init__()

def label_smoothing_loss(y_pred, y, dimension):
    loss = -(F.log_softmax(y_pred, dim=dimension) * y).sum()
    return loss
