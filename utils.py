import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sin, cos

def attnMask(energy, batch_lengths):
    masked_energy = energy.clone()
    for i, length in enumerate(batch_lengths):
            masked_energy[i, :, length:] = -np.inf
    return masked_energy

def outputMask(x, batch_lengths, max_len):
    mask = torch.ones_like(x)
    for i, length in enumerate(batch_lengths):
        if length < max_len:
            mask[i, -(max_len-length):, :] = 0
    return mask

def positionalEncoding(x):
        PE = torch.zeros_like(x[0])
        for pos in range(x.size(1)):
            for i in range(x.size(2)//2):
                PE[pos, 2*i] = sin(pos / 10000**(2*i/x.size(2)))
                PE[pos, 2*i+1] = cos(pos / 10000**(2*i/x.size(2)))
        return x + PE