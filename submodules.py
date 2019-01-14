import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import attnMask, outputMask
import numpy as np

    
class Head(nn.Module):
    def __init__(self, embedding_dim, dk, mode):
        super(Head, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        self.mode = mode
        
        self.Wq = nn.Linear(embedding_dim, dk, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk, bias=False)
        
    def forward(self, x, context, batch_lengths, max_len):
        mask = outputMask(x[:,:,:self.dk], batch_lengths, max_len)
        Q, K = self.Wq(context)*mask, self.Wk(context)*mask
        V = self.Wv(x)*mask
        
        energy = Q.bmm(K.permute(0,2,1))
        energy = attnMask(energy, batch_lengths)
        
        if self.mode=='Decoder':
            for i in range(energy.size(1)):
                for j in range(i+1, energy.size(2)):
                    energy[:,i,j] = -np.inf
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)
        return score.bmm(V)


class Layer(nn.Module):
    def __init__(self, embedding_dim, num_heads, mode):
        super(Layer, self).__init__()
        # Parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dk = embedding_dim // num_heads
        self.mode = mode
        
        # Layers
        if mode=='Decoder':
            self.Masked_attnHeads = nn.ModuleList([Head(embedding_dim, self.dk, 'Decoder') for _ in range(num_heads)])
            self.LN1 = nn.LayerNorm(embedding_dim)
        
        self.attnHeads = nn.ModuleList([Head(embedding_dim, self.dk, 'Encoder') for _ in range(num_heads)])
        self.LN2 = nn.LayerNorm(embedding_dim)
        
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        self.LN3 = nn.LayerNorm(embedding_dim)
        
        
    def forward(self, x, context, batch_lengths, max_len):
        mask = outputMask(x, batch_lengths, max_len)
        
        if self.mode=='Decoder':
            z = torch.cat( [ self.Masked_attnHeads[i](x, x, batch_lengths, max_len) for i in range(self.num_heads) ], dim=-1 ) * mask
            x = self.LN1(x+z)
        
        # if self.mode == 'Decoder' => context = x
        z = torch.cat( [ self.attnHeads[i](x, context, batch_lengths, max_len) for i in range(self.num_heads) ], dim=-1 ) * mask
        x = self.LN2(x+z)
        
        z = F.relu(self.FC1(x)) * mask
        z =self.FC2(z) * mask
        
        output = self.LN3(x+z)
        
        return output
    
    
class Attention(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scoring = nn.Linear(embedding_dim + 3*hidden_dim, 1)
    
    def forward(self, context, context_lengths, x, h):
        h = h.view(-1, 1, self.hidden_dim)
        inputs = torch.cat([x,h], dim=-1).repeat(1, context.size(1), 1)

        scores = self.scoring(torch.cat([inputs, context], dim=-1))
        for b, t in enumerate(context_lengths):
            scores[b, t:] = -np.inf
            
        alpha = F.softmax(scores, dim=1)

        return torch.sum( (alpha * context), dim=1 ).unsqueeze(1)
        
        
    
    
    
    
    
    
    
    
    
    