"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super(CrossEntropyLoss, self).__init__()
        if config == None:
            self.loss = nn.CrossEntropyLoss()
        else:
            class_weights = np.load(config.class_weights)
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index,
                                      weight=torch.from_numpy(class_weights.astype(np.float32)),
                                      size_average=True, reduce=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        if weight is not None:
            # print("weight is", weight)
            return F.cross_entropy(input, target, weight=weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            return F.cross_entropy(input, target, weight=self.weight,
                                   ignore_index=self.ignore_index, reduction=self.reduction)
