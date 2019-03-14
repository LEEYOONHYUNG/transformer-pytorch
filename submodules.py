import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import attnMask

INF = float('inf')

#################################################################################################################
##################################################   ENCODER   ##################################################
#################################################################################################################

class multiHead_enc(nn.Module):
    def __init__(self, embedding_dim, dk, num_heads):
        super(multiHead_enc, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        
        self.Wq = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        
    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        
        energy = Q.bmm(K.permute(0,2,1))
        
        for batch in energy:
            length = torch.sum(batch[0]!=0, dim=-1)
            batch[:, length:] = -INF
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)

        return score.bmm(V)


class encoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(encoderLayer, self).__init__()
        # Parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dk = embedding_dim // num_heads

        self.Heads = multiHead_enc(embedding_dim, self.dk, num_heads)
        self.LN1 = nn.LayerNorm(embedding_dim)
        
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        self.LN2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        z = self.Heads(x)
        x = self.LN1(x+z)
        
        z = F.relu(self.FC1(x))
        z = self.FC2(z)
        output = self.LN2(x+z)
        
        return output
    


#################################################################################################################
##################################################   DECODER   ##################################################
#################################################################################################################
class multiHead_att(nn.Module):
    def __init__(self, embedding_dim, dk, num_heads):
        super(multiHead_att, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        
        self.Wq = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        
    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        
        energy = Q.bmm(K.permute(0,2,1))
        lengths = torch.sum(energy[:,0]!=0, dim=-1)
        
        for i, batch in enumerate(energy):
            batch[:, lengths[i]:] = -INF
            
        for i in range(len(energy[0])):
            energy[:, i, i+1:] = -INF
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)
        
        return score.bmm(V) 
    

class multiHead_dec(nn.Module):
    def __init__(self, embedding_dim, dk, num_heads):
        super(multiHead_dec, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        
        self.Wq = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk * num_heads, bias=False)
        
    def forward(self, x, context):
        Q = self.Wq(x)
        K, V = self.Wk(context), self.Wv(context)
        
        energy = Q.bmm(K.permute(0,2,1))
        for batch in energy:
            length = torch.sum(batch[0]!=0, dim=-1)
            batch[:, length:] = -INF
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)
        
        return score.bmm(V)


    
class decoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(decoderLayer, self).__init__()
        # Parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dk = embedding_dim // num_heads
        
        self.attnHeads = multiHead_att(embedding_dim, self.dk, num_heads)
        self.LN1 = nn.LayerNorm(embedding_dim)

        self.decoderHeads = multiHead_dec(embedding_dim, self.dk, num_heads)
        self.LN2 = nn.LayerNorm(embedding_dim)
        
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        self.LN3 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, context):
        z = self.attnHeads(x)
        x = self.LN1(x+z)
        
        z = self.decoderHeads(x, context)
        x = self.LN2(x+z)
        
        z = F.relu(self.FC1(x))
        z = self.FC2(z)
        
        output = self.LN3(x+z)
        
        return output
    
    
    

    

        
        
    
    
    
    
    
    
    
    
    
    