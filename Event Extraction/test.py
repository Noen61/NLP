import numpy as np
from pytorch_pretrained_bert import BertModel,BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import json
from const import NONE,PAD,UNK,CLS,SEP,TRIGGERS,ARGUMENTS
from dataset import TrainDataset
from model import Net
from utils import build_vocab,find_triggers,calc_metric,Trainpad,tokenizer,all_triggers, trigger2idx, idx2trigger,all_arguments, argument2idx, idx2argument
import os

def eval(model, iterator):
    model.eval()
    scores = 0
    num_preds = 0
    num_trues = 0
    f1 = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            arguments_all, arguments_hat_all = [], []
            tokens_x_2d,triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, mask, words_2d, triggers_2d ,token_type_ids= batch
            trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d= model.predict_triggers(tokens_x_2d=tokens_x_2d,mask=mask,head_indexes_2d=head_indexes_2d,arguments_2d=arguments_2d,Test=True)
                
            arguments_all.extend(arguments_2d)
            arguments_hat_all.extend(argument_hat_2d)

            for _, (arguments, arguments_hat) in enumerate(zip(arguments_all, arguments_hat_all)):
                arguments_true, arguments_pred = [], [] 
                triggers_true, triggers_pred = [], [] 
                for trigger in arguments:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments[trigger]:
                        a_start, a_end, a_type_role = argument
                        arguments_true.append((t_type_str,a_type_role,a_start, a_end))
                for trigger in arguments_hat:
                    t_start, t_end, t_type_str = trigger
                    if len(arguments_hat[trigger])>0:
                        for argument in arguments_hat[trigger]:
                            a_start, a_end, a_type_role = argument
                            arguments_pred.append((t_type_str,a_type_role,a_start, a_end))
                score = calc_metric(arguments_true, arguments_pred)
                scores += score 
                num_preds += len(arguments_pred) 
                num_trues += len(arguments_true)
    
    if num_preds>0:           
        p = scores/num_preds
        r = scores/num_trues
        if p!=0 and r!=0:
            f1 = (2*p*r)/(p+r)
    return f1

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    model = Net(device=device,trigger_size=len(all_triggers),argument_size=len(all_arguments)).to(device)
    model.load_state_dict(torch.load('eemodels/best_ner_argument.pth'))
    test_dataset = TrainDataset('data/test.json')
    test_loader = data.DataLoader(dataset=test_dataset,batch_size=12,num_workers=2,collate_fn=Trainpad)
    argument_f1 = eval(model,test_loader)
    print(argument_f1)