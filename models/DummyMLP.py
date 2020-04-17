# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:20:39 2020

@author: GK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyMLP(nn.Module):
    """
    Just dumb fixed sized concatenating MLP, as baseline
    """
    def __init__(self, seq_length, input_size):
        super(DummyMLP, self).__init__()
        #Params
        self.seq_length = seq_length #number of timesteps T i input [fixed value for this kind of attention]
        self.input_size = input_size #dimension of each input vector (=number of features)
        #Layers
        self.MLP_block1 = nn.Sequential(
            nn.Linear(self.seq_length*self.input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
            )

    def forward(self, x):
        return self.MLP_block1(x)