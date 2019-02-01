import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils import attnMask, positionalEncoding
from math import sin, cos
from submodules import encoderLayer, decoderLayer


class Transformer_encoder(nn.Module):
    def __init__(self, num_layers, num_heads, voca_size, embedding_dim, embedding_matrix=None):
        super(Transformer_encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        
        if embedding_matrix is None:
            self.Embedding = nn.Embedding(voca_size, embedding_dim)
        else:
            self.Embedding = nn.Embedding.from_pretrained(embedding_matrix)
            
        self.Layers = nn.Sequential(*[encoderLayer(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        mask = (x!=0).unsqueeze(-1).type(torch.float)
        x = self.Embedding(x)
        x = positionalEncoding(x) * mask
        x = self.Layers(x)

        return x
        
        
        
        
class Transformer_decoder(nn.Module):
    def __init__(self, num_layers, num_heads, voca_size, embedding_dim, max_len):
        super(Transformer_decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.voca_size = voca_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        self.Embedding = nn.Embedding(voca_size, embedding_dim)
        self.Layers = nn.ModuleList([decoderLayer(embedding_dim, num_heads) for _ in range(num_layers)])
        self.output2word = nn.Linear(embedding_dim, voca_size)
        

        
    def forward(self, x, context, train=True):
        if train==True:
            mask = (x!=0).unsqueeze(-1).type(torch.float)
            x = self.Embedding(x)
            x = positionalEncoding(x) * mask

            for layer in self.Layers:
                x = layer(x, context)
            
            return self.output2word(x)
        
        else:
            input = x
            outputs = []
            
            for t in range(self.max_len):
                mask = (input!=0).unsqueeze(-1).type(torch.float)
                x = self.Embedding(input)
                x = positionalEncoding(x) * mask

                for layer in self.Layers:
                    x = layer(x, context)

                output = x[0:1, t:t+1]
                
                word = self.output2word(output)
                outputs.append(word)
                
                next_token = torch.argmax(word, dim=-1).item() # 다시 (1,1)
                
                if (next_token == 3) or (next_token==0): # <eos>: 3  <pad>:0
                    break
                    
                if (t+1)< self.max_len:
                    input[0,t+1] = next_token

            return torch.cat(outputs, dim=1)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        