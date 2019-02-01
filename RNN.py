import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import copy
from utils import attnMask, positionalEncoding, sort_batch, restore_batch
from math import sin, cos
from submodules import Attention
from utils import *





class RNN_encoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, voca_size, embedding_dim, embedding_matrix=None):
        super(RNN_encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        
        if embedding_matrix is None:
            self.Embedding = nn.Embedding(voca_size, embedding_dim)
        else:
            self.Embedding = nn.Embedding.from_pretrained(embedding_matrix)
            
        self.Encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        sorted_x, sorted_lengths, sorted_indices = sort_batch(x)
        sorted_x = self.Embedding(sorted_x)
        
        packed_inputs =  pack_padded_sequence(sorted_x, sorted_lengths.tolist(), batch_first=True)
        packed_outputs, _ = self.Encoder(packed_inputs)
        context, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        context = restore_batch(context, sorted_indices)
        
        return context

        

class RNN_decoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, voca_size, embedding_dim, max_len):
        super(RNN_decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        self.Embedding = nn.Embedding(voca_size, embedding_dim)
        self.Decoder = nn.GRU(embedding_dim + 2*hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.Attention = Attention(embedding_dim, hidden_dim)
        self.output2word = nn.Linear(hidden_dim, voca_size)
        
        
    def forward(self, x, context, train=True):
        outputs = []
        
        if train==True:
            h = torch.zeros( self.num_layers, x.size(0), self.hidden_dim)
            
            for t in range(x.size(1)):
                next_input = x[:, t:t+1] # 여기에는 pad가 섞여 있다. (B, 1)
                embedded_input = self.Embedding(next_input) # (B, 1, E)
                attenton_vector = self.Attention(context, embedded_input, h[0:1]) # (B, 1, 2h), first layer hidden state
                
                concat_input = torch.cat([embedded_input, attenton_vector], dim=-1) # (B, 1, E+2h) 
                output, h = self.Decoder(concat_input, h) # (B, 1, h) (1, B, h)

                output = output * ((next_input!=0).type(torch.float)).unsqueeze(-1) # (B, 1, 2h) (B, 1, 1)
                word = self.output2word(output)
                outputs.append(word)
                
        else:
            next_input = x # x.size() = (1,1)
            h = torch.zeros(self.num_layers, 1, self.hidden_dim)
            
            for t in range(self.max_len):
                embedded_input = self.Embedding(next_input)
                attenton_vector = self.Attention(context, embedded_input, h[0:1])
                
                concat_input = torch.cat([embedded_input, attenton_vector], dim=-1)
                output, h = self.Decoder(concat_input, h)
                
                word = self.output2word(output)
                outputs.append(word)
                
                next_input = torch.argmax(word, dim=-1) # 다시 (1,1)
                if (next_input.item() == 3) or (next_input.item()==0): # <eos>: 3  <pad>:0
                    break
                
        
        return torch.cat(outputs, dim=1)
        
