import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.w11 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.a11 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.h11 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h31 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.V1 = nn.Linear(HIDDEN2_UNITS, action_size)

        self.w12 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.a12 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.h12 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h32 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.V2 = nn.Linear(HIDDEN2_UNITS, action_size)


    def forward(self, s, a):
        w11 = F.relu(self.w11(s))
        a11 = self.a11(a)
        h11 = self.h11(w11)
        h21 = h11 + a11
        h31 = F.relu(self.h31(h21))
        out1 = self.V1(h31)

        w12 = F.relu(self.w12(s))
        a12 = self.a12(a)
        h12 = self.h12(w12)
        h22 = h11 + a12
        h32 = F.relu(self.h32(h22))
        out2 = self.V2(h32)
        return out1, out2

    def Q1(self, s, a):
        w11 = F.relu(self.w11(s))
        a11 = self.a11(a)
        h11 = self.h11(w11)
        h21 = h11 + a11
        h31 = F.relu(self.h31(h21))
        out1 = self.V1(h31)
        return out1