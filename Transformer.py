import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils import attnMask, outputMask, positionalEncoding
from math import sin, cos
from submodules import Layer


class Transformer_encoder(nn.Module):
    
    def __init__(self, num_layers, num_heads, voca_size, embedding_dim):
        super(Transformer_encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        
        self.Embedding = nn.Embedding(voca_size, embedding_dim)
        self.Layers = nn.ModuleList([Layer(embedding_dim, num_heads, 'Encoder') for _ in range(num_layers)])

    def forward(self, x, batch_lengths=None):
        batch_lengths = np.ones(x.size(0), dtype=np.int32) * x.size(1) if batch_lengths is None \
                        else np.asarray(batch_lengths).astype(int)
        max_len = x.size(1)
        
        x = self.Embedding(x)
        mask = outputMask(x, batch_lengths, max_len)
        x = positionalEncoding(x) * mask

        for layer in self.Layers:
            x = layer(x, x, batch_lengths, max_len)
            
        return x
        
        

class Transformer_decoder(nn.Module):
    
    def __init__(self, num_layers, num_heads, voca_size, embedding_dim):
        super(Transformer_decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        self.Embedding = nn.Embedding(voca_size, embedding_dim)
        self.Layers = nn.ModuleList([Layer(embedding_dim, num_heads, 'Decoder') for _ in range(num_layers)])
        self.energy2voca = nn.Linear(hidden_dim, voca_size)

        
        
    def forward(self, x, context, batch_lengths=None, max_len=None):
        batch_lengths = np.ones(x.size(0), dtype=np.int32) * x.size(1) if batch_lengths is None \
                        else np.asarray(batch_lengths).astype(int)
        max_len = x.size(1)
        
        x = self.Embedding(x)
        mask = outputMask(x, batch_lengths, max_len)
        x = positionalEncoding(x) * mask
        
        for layer in self.Layers:
            x = layer(x, context, batch_lengths, max_len)
            
        return self.energy2voca(x)
    
    
    
    def inference(self, context, batch_lengths=None, max_len=None):
        batch_lengths = np.ones(context.size(0), dtype=np.int32) * context.size(1) if batch_lengths is None else np.asarray(batch_lengths).astype(int)
        max_len = batch_lengths.max() if max_len is None else max_len
        
        decoder_inputs = torch.zeros(context.size(0), max_len, self.embedding_dim)
        outputs = torch.empty
        
        for t in range(max_len):
            energy = self.forward(decoder_inputs, context, batch_lengths, max_len)[:,t,:]
            output = self.energy2voca(energy)
            outputs.append(output)
            decoder_inputs[:,t+1,:] = output
            
        return torch.cat(outputs, dim=1)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        