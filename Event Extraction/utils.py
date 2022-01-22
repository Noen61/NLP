import numpy as np
from pytorch_pretrained_bert import BertModel,BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import json
from const import NONE,PAD,UNK,CLS,SEP,TRIGGERS,ARGUMENTS

def build_vocab(labels, BIO_tagging=True):
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label

def Trainpad(batch):
    tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, mask, words_2d, triggers_2d,token_type_ids = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        mask[i] = mask[i] + [0] * (maxlen - len(mask[i]))
        token_type_ids[i] = token_type_ids[i] + [0] * (maxlen - len(token_type_ids[i]))

    return tokens_x_2d,triggers_y_2d, arguments_2d,seqlens_1d, head_indexes_2d, mask, words_2d, triggers_2d,token_type_ids


def find_triggers(labels):
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]

def calc_metric(y_true, y_pred):
    num_pred = len(y_pred)     #预测论元数量
    num_true = len(y_true)     #人工标注论元数量
    #item:(t_type_str,a_type_role,a_start, a_end)
    y_true_set = [item[0:2] for item in y_true]
    score = 0
    if num_true>0:
        for item in y_pred:
            if item[0:2] in y_true_set:
                tmp = y_true[y_true_set.index(item[0:2])]
                num = min(item[3],tmp[3])-max(item[2],tmp[2])
                f1 = 0
                if num > 0:
                    p = num / (item[3] - item[2])
                    r = num / (tmp[3] - tmp[2])
                    f1 = (2*p*r)/(p+r)
                score += f1
    return score


tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-chinese-vocab.txt', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS)