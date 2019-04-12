import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter
from nltk import word_tokenize
import numpy as np
import re
from utils import clean_text
import copy

TEXT_SAMPLE = list(range(0, 150000, 15))
    
class TextPairs(Dataset):
    def __init__(self, num_voca, train=True, toy=True):
        super(TextPairs, self).__init__()
        self.dataset = {'en':[], 'de':[]}
        self.voca = {'en':[], 'de':[]}
        self.word2id = {'en':None, 'de':None}
        self._len = 0
        self.max_len = 0
        
        with open('Data/en-de.txt', 'r') as f:
            lines = f.read().splitlines()
            for i,line in enumerate(lines):
                if toy==True:
                    if i not in TEXT_SAMPLE:
                        continue
                        
                text_en, text_de = clean_text(line).split('\t')
                token_en, token_de = word_tokenize(text_en.strip()), word_tokenize(text_de.strip())
                
                self.dataset['en'].append(token_en)
                self.dataset['de'].append(token_de)
                
                if len(token_en) > self.max_len: 
                    self.max_len = len(token_en)
                    
                if len(token_de)+1 > self.max_len: 
                    self.max_len = len(token_de)+1
                    
                self.voca['en'].extend(token_en)
                self.voca['de'].extend(token_de)
                self._len += 1
            
        self.voca['en'] = list(list(zip(*(Counter(self.voca['en']).most_common(num_voca-4))))[0])
        self.voca['de'] = list(list(zip(*(Counter(self.voca['de']).most_common(num_voca-4))))[0])
        
        self.voca['en'] = ['<pad>', '<unk>', '<sos>', '<eos>'] +  self.voca['en']
        self.voca['de'] = ['<pad>', '<unk>', '<sos>', '<eos>'] +  self.voca['de']
        
        self.word2id['en'] = dict( [ (w, i) for i, w in enumerate(self.voca['en'])] )
        self.word2id['de'] = dict( [ (w, i) for i, w in enumerate(self.voca['de'])] )
        
        self.hyp2ref = dict([ (' '.join([t if t in self.voca['en'] else '<unk>' for t in k]), []) for k in self.dataset['en']])
        
        for k, v in zip(self.dataset['en'], self.dataset['de']):
            key = [t if t in self.voca['en'] else '<unk>' for t in k]
            self.hyp2ref[' '.join(key)].append(v)
        
        if train==True:
            self.dataset['en'] =  self.dataset['en'][:int(self._len*0.9)]
            self.dataset['de'] =  self.dataset['de'][:int(self._len*0.9)]
            self._len = len(self.dataset['en'])
            
        else:
            self.dataset['en'] =  self.dataset['en'][int(self._len*0.9):]
            self.dataset['de'] =  self.dataset['de'][int(self._len*0.9):]
            self._len = len(self.dataset['en'])
            
        
        
    def __len__(self):
        return self._len
        
        
    def __getitem__(self, idx):
        en_text = torch.zeros(self.max_len, dtype=torch.long)
        de_text = torch.zeros(self.max_len+1, dtype=torch.long)
        
        en = self.text2id( self.dataset['en'][idx], 'en' )
        de = self.text2id( self.dataset['de'][idx], 'de' )
        
        en_text[:len(en)] = en
        de_text[:len(de)] = de
        
        return {'en':en_text, 'de':de_text}
        
        
    def text2id(self, text, language):
        if language=='en':
            return torch.Tensor([ self.word2id[language].get(w) if self.word2id[language].get(w) is not None else 1 for w in text ])
        elif language=='de':
            return torch.Tensor([2] + [ self.word2id[language].get(w) if self.word2id[language].get(w) is not None else 1 for w in text ] + [3])

        
        
