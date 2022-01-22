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
from dataset import SemevalDataset
from model import MTB


def train(model, iterator, optimizer):
    model.train()
    loss = 0
    for i, batch in enumerate(iterator):
        tokens_x_2d, seqlens_1d,mask,e1_heads,e2_heads,relations = batch
        optimizer.zero_grad()
        re_loss,relations_pred= model.find_relation(tokens_x_2d=tokens_x_2d,mask=mask,e1_heads=e1_heads,e2_heads=e2_heads,relations=relations,Test=False)
        #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss += re_loss.item()
        re_loss.backward()
        optimizer.step()
        if i % 200 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss/(i+1)))
            
def eval(model, iterator):
    model.eval()
    trues=[]
    preds=[]
    for i, batch in enumerate(iterator):
        tokens_x_2d, seqlens_1d,mask,e1_heads,e2_heads,relations = batch
        re_loss,relations_pred= model.find_relation(tokens_x_2d=tokens_x_2d,mask=mask,e1_heads=e1_heads,e2_heads=e2_heads,relations=relations,Test=True)
        preds.extend(relations_pred.detach().cpu().tolist())
        trues.extend(relations)
    return f1_score(trues,preds,average='macro')

if __name__ == '__main__':
    train_path = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    test_path = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    train_dataset = SemevalDataset(train_path,'train')
    train_loader = DataLoader(dataset=train_dataset,batch_size=10,shuffle=True,num_workers=2,collate_fn=Trainpad)
    test_dataset = SemevalDataset(test_path,'test')
    test_loader =DataLoader(dataset=test_dataset,batch_size=10,num_workers=2,collate_fn=Trainpad)
    device=torch.device("cpu")
    if torch.cuda.is_available():
        device = 'cuda'
    model = MTB(relation_size=len(id2re),device=device).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.00002)

    early_stop = 15
    stop = 0
    best_scores = 0.0
    for epoch in range(100):
            print("=========train at epoch:{}=========".format(epoch))
            train(model,train_loader, optimizer)
            print("=========dev at epoch:{}=========".format(epoch))
            f1 = eval(model, test_loader)
            print(f1)
            print("=========save at epoch:{}=========".format(epoch))
            if stop >= early_stop:
                print("The best result in epoch:{}".format(epoch-early_stop-1))
                break
            if f1 > best_scores:
                best_scores = f1
                stop = 0
                print("The new best in epoch:{}".format(epoch))
                torch.save(model, "remodels/re.pth")