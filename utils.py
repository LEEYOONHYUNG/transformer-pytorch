import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from math import sin, cos
import re

INF = float('inf')

def positionalEncoding(x):
        PE = torch.zeros_like(x[0])
        for pos in range(x.size(1)):
            for i in range(x.size(2)//2):
                PE[pos, 2*i] = sin(pos / 10000**(2*i/x.size(2)))
                PE[pos, 2*i+1] = cos(pos / 10000**(2*i/x.size(2)))
        return x + PE
    
def attnMask(energy, batch_lengths):
    masked_energy = energy.clone()
    for i, length in enumerate(batch_lengths):
            masked_energy[i, :, length:] = -INF
    return masked_energy


def sort_batch(batch):
    batch_lengths = torch.sum(batch!=0, dim=-1)
    sorted_lengths, sorted_indices = torch.sort(batch_lengths, descending=True)
    sorted_batch = batch[sorted_indices]
    
    return sorted_batch, sorted_lengths, sorted_indices
    
    
def restore_batch(sorted_batch, sorted_indices):
    restored_batch = torch.zeros_like(sorted_batch)
    
    for i, j in enumerate(sorted_indices):
        restored_batch[j] = sorted_batch[i]

    return restored_batch


def alignment(tensor, texts, device):
    max_len = max([tensor.size(1), texts.size(1)])
    
    pad_tensor = torch.zeros( len(tensor), max_len, tensor.size(2) )
    for i, data in enumerate(tensor):
        for j, row in enumerate(data):
            pad_tensor[i, j, :len(row)] = row
            
    pad_texts = torch.zeros( len(texts), max_len, dtype=torch.long )
    for i, text in enumerate(texts):
        pad_texts[i,:len(text)] = text

    return pad_tensor.to(device), pad_texts.to(device)
                  
             
        
        
        
        
        
        
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"i'm ", "i am ", text)
    text = re.sub(r"he's ", "he is ", text)
    text = re.sub(r"she's ", "she is ", text)
    text = re.sub(r"isn't ", "is not ", text)
    text = re.sub(r"aren't ", "are not ", text)
    text = re.sub(r"ain't ", "are not ", text)
    text = re.sub(r"it's ", "it is ", text)
    text = re.sub(r"that's ", "that is ", text)
    text = re.sub(r"what's ", "what is ", text)
    text = re.sub(r"where's ", "where is ", text)
    text = re.sub(r"who's ", "who is ", text)
    text = re.sub(r"how's ", "how is ", text)
    text = re.sub(r"\'ll ", " will ", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"\'re ", " are ", text)
    text = re.sub(r"\'d ", " would ", text)
    text = re.sub(r"\'re ", " are ", text)
    text = re.sub(r"won't ", "will not ", text)
    text = re.sub(r"can't ", "can not ", text)
    text = re.sub(r"couldn't ", "could not ", text)
    text = re.sub(r"shouldn't ", "should not ", text)
    text = re.sub(r"didn't ", "did not ", text)
    text = re.sub(r"[!]+", " ! ", text)
    text = re.sub(r"[\?]+", " ? ", text)
    text = re.sub(r"[\.]+", " . ", text)
    text = re.sub(r"[\,]+", " , ", text)
    text = re.sub(r"[\-‼!\.\(\’\[\]\)\"#\*/@�\^^;:<>{}`'\+=~|]", " ", text)
    text = re.sub(r"[\ ]+", " ", text)
    text = re.sub(r"don't ", "do not ", text)
    
    return text.strip()#맨 마지막 띄어쓰기 생략
    
    
    