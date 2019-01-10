import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils import attnMask, outputMask, positionalEncoding
from math import sin, cos
from submodules import encoderLayer, decoderLayer


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, embedding_dim):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        self.Layers = nn.ModuleList([encoderLayer(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, batch_lengths=None, max_len=None):
        batch_lengths = np.ones(x.size(0), dtype=np.int32) * x.size(1) if batch_lengths is None else np.asarray(batch_lengths).astype(int)
        max_len = batch_lengths.max() if max_len is None else max_len
        
        mask = outputMask(x, batch_lengths, max_len)
        x = positionalEncoding(x) * mask

        for layer in self.Layers:
            x = layer(x, batch_lengths, max_len)
            
        return x
        

class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, embedding_dim):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        self.Layers = nn.ModuleList([decoderLayer(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, context, batch_lengths=None, max_len=None):
        batch_lengths = np.ones(x.size(0), dtype=np.int32) * x.size(1) if batch_lengths is None else np.asarray(batch_lengths).astype(int)
        max_len = batch_lengths.max() if max_len is None else max_len
        
        mask = outputMask(x, batch_lengths, max_len)
        x = positionalEncoding(x) * mask
        
        for layer in self.Layers:
            x = layer(x, context, batch_lengths, max_len)
            
        return x