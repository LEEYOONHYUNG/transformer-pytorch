import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import attnMask

INF = float('inf')

#################################################################################################################
##################################################   ENCODER   ##################################################
#################################################################################################################

class encoderHead(nn.Module):
    def __init__(self, embedding_dim, dk):
        super(encoderHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        
        self.Wq = nn.Linear(embedding_dim, dk, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk, bias=False)
        
    def forward(self, x):
        mask = 1 - (x.eq(0).all(dim=2, keepdim=True).type(torch.float))
        Q, K, V = self.Wq(x)*mask, self.Wk(x)*mask, self.Wv(x)*mask
        
        energy = Q.bmm(K.permute(0,2,1))
        for batch in energy:
            length = torch.sum(batch[0]!=0, dim=-1)
            batch[:, length:] = -INF
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)

        return score.bmm(V) * mask


class encoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(encoderLayer, self).__init__()
        # Parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dk = embedding_dim // num_heads

        self.Heads = nn.ModuleList([encoderHead(embedding_dim, self.dk) for _ in range(num_heads)])
        self.LN1 = nn.LayerNorm(embedding_dim)
        
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        self.LN2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        mask = 1 - (x.eq(0).all(dim=2, keepdim=True).type(torch.float))
        z = torch.cat( [ self.Heads[i](x) for i in range(self.num_heads) ], dim=-1 )
        x = self.LN1(x+z)
        
        z = F.relu(self.FC1(x)) * mask
        z = self.FC2(z) * mask
        output = self.LN2(x+z)
        
        return output
    


#################################################################################################################
##################################################   DECODER   ##################################################
#################################################################################################################
class attnHead(nn.Module):
    def __init__(self, embedding_dim, dk):
        super(attnHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        
        self.Wq = nn.Linear(embedding_dim, dk, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk, bias=False)
        
    def forward(self, x):
        mask = 1 - (x.eq(0).all(dim=2, keepdim=True).type(torch.float))
        Q, K, V = self.Wq(x)*mask, self.Wk(x)*mask, self.Wv(x)*mask
        
        energy = Q.bmm(K.permute(0,2,1))
        for batch in energy:
            length = torch.sum(batch[0]!=0, dim=-1)
            batch[:, length:] = -INF
            
        for i in range(len(energy[0])):
            for j in range(i+1, len(energy[0])):
                energy[:,i,j] = -INF
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)
        
        return score.bmm(V) * mask
    

class decoderHead(nn.Module):
    def __init__(self, embedding_dim, dk):
        super(decoderHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.dk = dk
        
        self.Wq = nn.Linear(embedding_dim, dk, bias=False)
        self.Wk = nn.Linear(embedding_dim, dk, bias=False)
        self.Wv = nn.Linear(embedding_dim, dk, bias=False)
        
    def forward(self, x, context):
        mask = 1 - (x.eq(0).all(dim=2, keepdim=True).type(torch.float))
        Q = self.Wq(x)*mask
        K, V = self.Wk(context)*mask, self.Wv(context)*mask
        
        energy = Q.bmm(K.permute(0,2,1))
        for batch in energy:
            length = torch.sum(batch[0]!=0, dim=-1)
            batch[:, length:] = -INF
        
        score = F.softmax( energy / (self.dk**0.5), dim=-1)
        
        return score.bmm(V) * mask


    
class decoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(decoderLayer, self).__init__()
        # Parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dk = embedding_dim // num_heads
        
        self.attnHeads = nn.ModuleList([attnHead(embedding_dim, self.dk) for _ in range(num_heads)])
        self.LN1 = nn.LayerNorm(embedding_dim)

        self.decoderHeads = nn.ModuleList([decoderHead(embedding_dim, self.dk) for _ in range(num_heads)])
        self.LN2 = nn.LayerNorm(embedding_dim)
        
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.FC2 = nn.Linear(embedding_dim, embedding_dim)
        self.LN3 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, context):
        mask = 1 - (x.eq(0).all(dim=2, keepdim=True).type(torch.float))
        z = torch.cat( [ self.attnHeads[i](x) for i in range(self.num_heads) ], dim=-1 )
        x = self.LN1(x+z)
        
        z = torch.cat( [ self.decoderHeads[i](x, context) for i in range(self.num_heads) ], dim=-1 )
        x = self.LN2(x+z)
        
        z = F.relu(self.FC1(x)) * mask
        z = self.FC2(z) * mask
        
        output = self.LN3(x+z)
        
        return output
    
    
    
    
class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scoring = nn.Linear(embedding_dim + 3*hidden_dim, 1)
    
    def forward(self, context, x, h):
        h = h.view(-1, 1, self.hidden_dim) # (Batch_size, 1, hidden_dim)
        inputs = torch.cat([x,h], dim=-1).repeat(1, context.size(1), 1) # (Batch_size, max_len, embedding_dim + hidden_dim)

        scores = self.scoring(torch.cat([inputs, context], dim=-1)) # (Batch_size, max_len, embedding_dim + 3*hidden_dim) -> (Batch_size, max_len, 1)
        
        context_lengths = torch.sum((context!=0)[:,:,0], dim=-1).tolist()
        for i, length in enumerate(context_lengths):
            scores[i, length:] = -INF
            
        alpha = F.softmax(scores, dim=1) # (Batch_size, max_len, 1)

        # alpha: (Batch_size, max_len, 1), context: (Batch_size, max_len, 2*h)
        attention_vector = torch.sum( (alpha * context), dim=1 ).unsqueeze(1)
        
        return  attention_vector # (Batch_size, 1, 2*h)
        
        
    
    
    
    
    
    
    
    
    
    