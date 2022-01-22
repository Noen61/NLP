import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel,BertTokenizer
from torch.utils.data import Dataset, DataLoader
from const import id2re,re2id


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({'additional_special_tokens':['[unused0]', '[unused1]', '[unused2]', '[unused3]']})

def tokenize(text):
    tokenized_text = ['[CLS]']+tokenizer.tokenize(text)+ ['[SEP]']
    e1_head = tokenized_text.index('[unused0]')
    e1_tail = tokenized_text.index('[unused1]')
    e2_head = tokenized_text.index('[unused2]')
    e2_tail = tokenized_text.index('[unused3]')
    token_x = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    return token_x,e1_head,e1_tail,e2_head,e2_tail

def process_text(text, mode='train'):
    tokens_x,relations,e1_heads,e1_tails,e2_heads,e2_tails = [],[],[],[],[],[]
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        relation = re.sub(r'\n+', '', relation)
        # 检验句子序号是否正确
        if mode == 'train':
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
            
        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[unused0] ', sent)
        sent = re.sub('</e1>', ' [unused1]', sent)
        sent = re.sub('<e2>', '[unused2] ', sent)
        sent = re.sub('</e2>', ' [unused3]', sent)
        
        token_x,e1_head,e1_tail,e2_head,e2_tail = tokenize(sent)
        
        tokens_x.append(token_x); relations.append(re2id[relation])
        e1_heads.append(e1_head); e1_tails.append(e1_tail)
        e2_heads.append(e2_head); e2_tails.append(e2_tail)  
        
    return tokens_x,relations,e1_heads,e1_tails,e2_heads,e2_tails

def Trainpad(batch):
    tokens_x_2d,seqlens_1d,mask,e1_heads,e2_heads,relations = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        mask[i] = mask[i] + [0] * (maxlen - len(mask[i]))
        #token_type_ids[i] = token_type_ids[i] + [0] * (maxlen - len(token_type_ids[i]))

    return tokens_x_2d,seqlens_1d,mask,e1_heads,e2_heads,relations
