import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_prepare import TextPairs
import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader
from RNN import RNN_encoder, RNN_decoder
from Transformer import Transformer_encoder, Transformer_decoder
from datetime import datetime
from math import sin, cos
from nltk.translate.bleu_score import corpus_bleu

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', default=5, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=400, type=int, help='Batch size')
parser.add_argument('--voca-size', default=15000, type=int, help='Voca size')
parser.add_argument('--nlayer', default=4, type=int, help='Number of layers')
parser.add_argument('--nhead', default=8, type=int, help='Number of heads')
parser.add_argument('--edim', default=200, type=int, help='Embedding dimension')
parser.add_argument('--toy', default=False, type=bool, help='Toy data')
args = parser.parse_args()

# toy: 10,000 sentences / real: 176,692 sentences
# 4473 ,6706 / 15151, 32829
start = datetime.now()
print("Data preparing...")
train_pairs = TextPairs(args.voca_size, train=True, toy=args.toy)
val_pairs = TextPairs(args.voca_size, train=False, toy=args.toy)

print( len(train_pairs.voca['en']) )
print( len(train_pairs.voca['de']) )

trainLoader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, num_workers=4)
valLoader = DataLoader(val_pairs, num_workers=4)

MAX_LEN = train_pairs.max_len
SAMPLE = [15, 5015, 10015]
print(f'\nElapsed time: {datetime.now() - start}')
print(f'\nData_length: {len(train_pairs)}')

with open('Data/Glove/glove.6B.200d.pkl', 'rb') as f:
    glove = pkl.load(f)

embedding_matrix = torch.zeros(args.voca_size, args.edim)

for w in train_pairs.voca['en']:
    if glove.get(w) is None:
        embedding_matrix[ train_pairs.word2id['en'][w] ] = torch.zeros(args.edim)
    else:
        embedding_matrix[ train_pairs.word2id['en'][w] ] = torch.from_numpy(glove.get(w))
        
PE = torch.zeros(1, MAX_LEN, args.edim)
for pos in range(MAX_LEN):
    for i in range(args.edim//2):
        PE[0, pos, 2*i] = sin(pos / 10000**(2*i/args.edim))
        PE[0, pos, 2*i+1] = cos(pos / 10000**(2*i/args.edim))
        
class TF_MODEL(nn.Module):
    def __init__(self, NUM_LAYERS, NUM_HEADS, VOCA_SIZE, EMBEDDING_DIM, embedding_matrix, MAX_LEN):
        super(TF_MODEL, self).__init__()
        self.encoder = Transformer_encoder(NUM_LAYERS, NUM_HEADS, VOCA_SIZE, EMBEDDING_DIM, embedding_matrix)
        self.decoder = Transformer_decoder(NUM_LAYERS, NUM_HEADS, VOCA_SIZE, EMBEDDING_DIM, MAX_LEN)
        
    def forward(self, encoder_inputs, PE, decoder_inputs, train):
        context = self.encoder(encoder_inputs, PE)
        preds = self.decoder(decoder_inputs, context, PE, train) # BATCH_SIZE, MAX_LEN, hidden_dim
        return preds
    

model = TF_MODEL(args.nlayer, args.nhead, args.voca_size, args.edim, embedding_matrix, MAX_LEN)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')
model = model.to(gpu)

num_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {num_params}')

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
PE = PE.to(gpu)

print(datetime.now())

train_losses = []
val_losses = []
BLEUS = []
for epoch in range(args.nepoch):
    train_loss = 0
    print("\n\n")
    print(f'Epoch: {epoch+1}\t', datetime.now())
    
    for i, data in enumerate(trainLoader):
        en_text, de_text = data['en'], data['de']
        
        encoder_inputs, decoder_inputs, targets = en_text, de_text[:,:-1], de_text[:,1:]
        encoder_inputs = encoder_inputs.to(gpu)
        decoder_inputs = decoder_inputs.to(gpu)
        targets = targets.to(gpu)
        
        preds = model(encoder_inputs, PE, decoder_inputs, train=True) # BATCH_SIZE, MAX_LEN, hidden_dim
        loss = criterion( preds.view(-1, args.voca_size), targets.contiguous().view(-1))
        
        optimizer.zero_grad()
        
        loss.backward()
        train_loss += float(loss)/150

        optimizer.step()
        
        
        if (i+1) % 150==0: # len(trainLoader) = 398
            train_losses.append(train_loss)
            references, hypotheses = [], []
            val_loss = 0
            model = model.to(cpu)
            PE = PE.to(cpu)
            with torch.no_grad():
                for j, data in enumerate(valLoader):
                    en_text, de_text = data['en'], data['de']

                    sos = torch.cat( [torch.tensor([[2]]), torch.zeros(1, MAX_LEN-1, dtype=torch.long)], dim=-1 )
                    preds = model(en_text, PE, sos, train=False) # BATCH_SIZE, MAX_LEN, hidden_dim

                    preds_loss = preds.new_zeros(MAX_LEN, args.voca_size)
                    preds_loss[:len(preds[0])] = preds[0]
                    targets = de_text[:,1:]
                    loss = criterion( preds_loss, targets.contiguous().view(-1))
                    val_loss += float(loss)/len(valLoader)

                    tokens = torch.argmax(preds[0], dim=-1)
                    text = [ val_pairs.voca['de'][t] for t in tokens if t not in [0,2,3]]

                    reference = val_pairs.hyp2ref[ ' '.join([val_pairs.voca['en'][t] for t in en_text[0] if t not in [0,2,3]]) ]
                    hypothesis = text

                    references.append(reference)
                    hypotheses.append(hypothesis)

                    if j in SAMPLE:
                        print('Pred:\t', hypothesis)
                        print('Target:\t', reference[0])
                
                
            val_losses.append(val_loss)
            BLEUS.append(corpus_bleu(references, hypotheses))
            print(f'Train loss:\t{train_losses[-1]:.3f}')
            print(f'Val loss:\t{val_losses[-1]:.3f}')
            print('BLEU score:\t', BLEUS[-1])
            train_loss=0
            model = model.to(gpu)
            PE = PE.to(gpu)