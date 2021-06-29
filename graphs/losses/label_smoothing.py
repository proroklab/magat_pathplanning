'''
Label Smoothed version of CrossEntropy
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class LabelSmoothing(nn.Module):

    def __init__(self, size, smoothing=0.0):
        '''

        Args:
            size: size of input
            smoothing: smoothing effect from 0 to 0.5.
        '''
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False)

        # self.padding_idx = padding_idx

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing

        self.size = size

        self.true_dist = None

        self.Logsoftmax = nn.LogSoftmax()

    def forward(self, x, target):
        assert x.size(1) == self.size
        x = self.Logsoftmax(x)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))