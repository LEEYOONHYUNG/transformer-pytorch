import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import copy
from utils import attnMask, outputMask, positionalEncoding, sort_batch, restore_batch
from math import sin, cos
from submodules import Attention




class RNN_encoder(nn.Module):
    
    def __init__(self, num_layers, hidden_dim, voca_size, embedding_dim):
        super(RNN_encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        
        self.Embedding = nn.Embedding(voca_size, embedding_dim)
        self.Encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, batch_lengths=None):
        batch_lengths = [ x.size(1) for _ in range(x.size(0)) ] if batch_lengths is None else batch_lengths
        assert isinstance(batch_lengths, list), "batch_lengths should be a list"

        x = self.Embedding(x)
        x, batch_lengths, sorted_idx = sort_batch(x, batch_lengths)
        
        packed_inputs =  pack_padded_sequence(x, batch_lengths, batch_first=True)
        packed_outputs, h = self.Encoder(packed_inputs)
        context, context_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        
        context, context_lengths = restore_batch(context, context_lengths, sorted_idx)
        
        return context, context_lengths
        
        
        

class RNN_decoder(nn.Module):
    
    def __init__(self, num_layers, hidden_dim, voca_size, embedding_dim, max_len):
        super(RNN_decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        self.Embedding = nn.Embedding(voca_size, embedding_dim)
        self.Decoder = nn.GRU(embedding_dim + 2*hidden_dim, hidden_dim, batch_first=True)
        self.Attention = Attention(embedding_dim, hidden_dim)
        self.energy2voca = nn.Linear(hidden_dim, voca_size)
        
        
        
    def forward(self, x, context, context_lengths, batch_lengths=None, eos=3, train=True):
        batch_lengths = [ x.size(1) for _ in range(x.size(0)) ] if batch_lengths is None else batch_lengths
        assert isinstance(batch_lengths, list), "batch_lengths should be a list"
        
        
        outputs = []
        
        if train:
            h = torch.zeros(1, x.size(0), self.hidden_dim)
            x = self.Embedding(x)
            
            for t in range(x.size(1)):
                next_input = x[:, t:t+1] # 여기에는 pad가 섞여 있다. (B, 1, E)
                summary = self.Attention(context, context_lengths, next_input, h) # (B, 1, 2h)

                mask = (t < torch.tensor(batch_lengths)).type(torch.float)
                concat = torch.cat([next_input, summary], dim=-1) # (B, 1, E+2h) 
                concat = concat * (mask.view(-1,1,1))
                
                energy, h = self.Decoder(concat, h) # (B, 1, h) (1, B, h)
                energy = energy * (mask.view([-1,1,1]))
                h = h * (mask.view([1,-1,1]))

                output = self.energy2voca(energy)
                outputs.append(output)
                
        else:
            next_input = x # x.size() = (1,1)
            h = torch.zeros(1, 1, self.hidden_dim)
            
            for t in range(self.max_len):
                next_input = self.Embedding(next_input)
                summary = self.Attention(context, context_lengths, next_input, h)
                
                concat = torch.cat([next_input, summary], dim=-1)
                energy, h = self.Decoder(concat, h)
                outputs.append(energy)
                
                voca = self.energy2voca(energy)
                next_input = torch.argmax(voca, dim=-1) # 다시 (1,1)
                
                if voca == eos: break
                
        return torch.cat(outputs, dim=1)
        
        
        
        
        
        
        
        
        
        
        
        