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

class MTB(nn.Module):
    def __init__(self,relation_size, device=torch.device("cpu")):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.resize_token_embeddings(len(tokenizer))
        hidden_size = 768
        self.fc = nn.Linear(2*hidden_size,relation_size)
        self.device = device
        self.relation_size = relation_size
        
    def find_relation(self,tokens_x_2d,mask,e1_heads,e2_heads,relations=None,Test=False): 
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        mask = torch.Tensor(mask).to(self.device)
        e1_heads = torch.LongTensor(e1_heads).to(self.device)
        e2_heads = torch.LongTensor(e2_heads).to(self.device)
        relations = torch.LongTensor(relations).to(self.device)
        hiddens=self.bert(tokens_x_2d,mask)
        x=hiddens[0]
        batch = len(tokens_x_2d)
        entity_marks = []
        loss = 0
        relations_pred = []
        for i in range(batch):
            entity_mark = torch.cat((x[i,e1_heads[i],:],x[i,e2_heads[i],:])).unsqueeze(0)
            entity_marks.append(entity_mark)
        entity_marks = torch.cat(entity_marks,dim=0) #(batch,2*hidden)
        logits = self.fc(entity_marks) #(batch,relations)
        if Test==False:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits,relations)
        if Test==True:
            relations_pred = F.softmax(logits,dim=-1).argmax(dim=-1) #(batch)
        
        return loss,relations_pred