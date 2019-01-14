import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
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
    
    
    
def sort_batch(batch, batch_lengths):
    batch_lengths = torch.LongTensor(batch_lengths)
    lengths, sorted_idx = batch_lengths.sort(0, descending=True)
    ordered_batch = batch[sorted_idx]
    
    return ordered_batch, lengths, sorted_idx
    
    
    
def restore_batch(batch, context_lengths, sorted_idx):
    restored_batch = batch.clone()
    restored_lengths = [ 0 for _ in context_lengths ]
    
    for i, j in enumerate(sorted_idx):
        restored_batch[j] = batch[i]
        restored_lengths[j] = context_lengths[i]

    return restored_batch, restored_lengths
    
    
    