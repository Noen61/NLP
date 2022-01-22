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
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils import tokenizer,tokenize,process_text,Trainpad
from const import id2re,re2id

class SemevalDataset(Dataset):
    def __init__(self,path,mods):
        self.tokens_x,self.relations,self.e1_heads,self.e1_tails,self.e2_heads,self.e2_tails=[],[],[],[],[],[]
        with open(path, 'r', encoding='utf8') as f:
            text = f.readlines()
        tokens_x,relations,e1_heads,e1_tails,e2_heads,e2_tails = process_text(text,mods)
        self.tokens_x = tokens_x
        self.relations = relations
        self.e1_heads = e1_heads
        self.e1_tails = e1_tails
        self.e2_heads = e2_heads
        self.e2_tails = e2_tails
    
    def __len__(self):
        return len(self.tokens_x)
        
    def __getitem__(self,idx):
        seqlen = len(self.tokens_x[idx])
        mask = [1] * seqlen
        return self.tokens_x[idx],seqlen,mask,self.e1_heads[idx],self.e2_heads[idx],self.relations[idx]