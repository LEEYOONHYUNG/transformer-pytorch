import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter
from nltk import word_tokenize
import numpy as np
import re
from utils import clean_text

TEXT_SAMPLE = list(range(100000, 150000, 1000))

    
class TextPairs(Dataset):
    def __init__(self, num_voca):
        super(TextPairs, self).__init__()
        self.dataset = {'en':[], 'de':[]}
        self.voca = {'en':[], 'de':[]}
        self.word2id = {'en':None, 'de':None}
        self._len = 0
        self.max_len = 0
        
        with open('Data/en-de.txt', 'r') as f:
            for i,line in enumerate(f):
                if i not in TEXT_SAMPLE:
                    continue
                text_en, text_de = line.split('\t')
                text_en, text_de = word_tokenize(clean_text(text_en)), word_tokenize(clean_text(text_de))
                self.dataset['en'].append(text_en)
                self.dataset['de'].append(text_de)
                
                if len(text_en) > self.max_len: 
                    self.max_len = len(text_en)
                    
                if len(text_de)+2 > self.max_len: 
                    self.max_len = len(text_de)+2
                    
                self.voca['en'].extend(text_en)
                self.voca['de'].extend(text_de)
                self._len += 1
            
                    
        self.voca['en'] = list(list(zip(*(Counter(self.voca['en']).most_common(num_voca-4))))[0])
        self.voca['de'] = list(list(zip(*(Counter(self.voca['de']).most_common(num_voca-4))))[0])
        
        self.voca['en'] = ['<pad>', '<unk>', '<sos>', '<eos>'] +  self.voca['en']
        self.voca['de'] = ['<pad>', '<unk>', '<sos>', '<eos>'] +  self.voca['de']
        
        self.word2id['en'] = dict( [ (w, i) for i, w in enumerate(self.voca['en'])] )
        self.word2id['de'] = dict( [ (w, i) for i, w in enumerate(self.voca['de'])] )
        
        
    def __len__(self):
        return self._len
        
    def __getitem__(self, idx):
        eng_text = self.text2id( self.dataset['en'][idx], 'en' )
        de_text = self.text2id( self.dataset['de'][idx], 'de' )
        return eng_text, de_text
        
    def text2id(self, text, language):
        if language=='en':
            return [ self.word2id[language].get(w) if self.word2id[language].get(w) is not None else 1 for w in text ]
        elif language=='de':
            return [2] + [ self.word2id[language].get(w) if self.word2id[language].get(w) is not None else 1 for w in text ] + [3]

        
        

def padding(texts):
    max_len = max(map(len, texts))
    padded_text=  torch.zeros(len(texts), max_len, dtype = torch.long)
    
    for i, text in enumerate(texts):
        padded_text[i, :len(text)] = torch.LongTensor(text)
    
    return padded_text




def pad_batch(batch):
    eng_text, de_text = list(zip(*batch))
    return padding(eng_text), padding(de_text)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        