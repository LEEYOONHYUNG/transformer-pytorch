import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import attnMask, outputMask
import numpy as np

    
class Head(nn.Module):
    def __init__(self, embedding_dim, dk):
        super(Head, self).__init__()
        self.dk = dk
        self.Wq = nn.Linear(embedding_dim, dk, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk, bias=False)
        
    def forward(self, x, context, batch_lengths, max_len, decoding=False):
        mask = outputMask(x[:,:,:self.dk], batch_lengths, max_len)
        Q, K = self.Wq(context)*mask, self.Wk(context)*mask
        V = self.Wv(x)*mask
        
        energy = Q.bmm(K.permute(0,2,1))
        energy = attnMask(energy, batch_lengths)
        
        if decoding:
            for i in range(energy.size(1)):
                for j in range(i+1, energy.size(2)):
                    energy[:,i,j] = -np.inf
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)
        return score.bmm(V)
    
    
class encoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(encoderLayer, self).__init__()
        self.dk = embedding_dim // num_heads
        self.num_heads = num_heads
        self.Heads = nn.ModuleList([Head(embedding_dim, self.dk) for _ in range(num_heads)])
        self.LN1 = nn.LayerNorm(embedding_dim)
        self.LN2 = nn.LayerNorm(embedding_dim)
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x, batch_lengths, max_len):
        mask = outputMask(x, batch_lengths, max_len)
        z = torch.cat( [ self.Heads[i](x, x, batch_lengths, max_len) for i in range(self.num_heads) ], dim=-1 ) * mask # Encoder: context = x
        x = self.LN1(x+z)
        z = F.relu(self.FC1(x)) * mask
        z =self.FC2(z) * mask
        
        output = self.LN2(x+z)
        
        return output
    
    
    
class decoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(decoderLayer, self).__init__()
        self.dk = embedding_dim // num_heads
        self.num_heads = num_heads
        self.Heads1 = nn.ModuleList([Head(embedding_dim, self.dk) for _ in range(num_heads)])
        self.Heads2 = nn.ModuleList([Head(embedding_dim, self.dk) for _ in range(num_heads)])
        self.LN1 = nn.LayerNorm(embedding_dim)
        self.LN2 = nn.LayerNorm(embedding_dim)
        self.LN3 = nn.LayerNorm(embedding_dim)
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x, context, batch_lengths, max_len):
        mask = outputMask(x, batch_lengths, max_len)

        z = torch.cat( [ self.Heads1[i](x, x, batch_lengths, max_len, decoding=True) for i in range(self.num_heads) ], dim=-1 ) * mask
        x = self.LN1(x+z)
        z = torch.cat( [ self.Heads2[i](x, context, batch_lengths, max_len) for i in range(self.num_heads) ], dim=-1 ) * mask
        x = self.LN2(x+z)
        z = F.relu(self.FC1(x)) * mask
        z =self.FC2(z) * mask

        output = self.LN3(x+z)
        
        return output