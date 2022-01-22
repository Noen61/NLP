import numpy as np
from pytorch_pretrained_bert import BertModel,BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import json
from const import NONE,PAD,UNK,CLS,SEP,TRIGGERS,ARGUMENTS
from utils import build_vocab,find_triggers,Trainpad,tokenizer,all_triggers, trigger2idx, idx2trigger,all_arguments, argument2idx, idx2argument


class TrainDataset(data.Dataset):
    def __init__(self,fpath):
        self.sent, self.id, self.triggers, self.arguments = [], [], [], []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                sentence = data['text'].replace(' ', '-')
                sentence = sentence.replace('\n', ',')
                sentence = sentence.replace('\u3000', '-')
                sentence = sentence.replace('\xa0', ',')
                sentence = sentence.replace('\ue627', ',')
                words = [word for word in sentence]
                if len(words) > 500:
                    continue
                triggers = [NONE] * len(words)
                arguments = {
                    # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                }
                for event_mention in data['event_list']:
                    id_s = event_mention['trigger_start_index']
                    trigger_text = event_mention['trigger']
                    id_e = id_s + len(trigger_text)
                    trigger_type = event_mention['event_type'].split('-')[-1]
                    for i in range(id_s, id_e):
                        if i == id_s:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)
                    event_key = (id_s, id_e, trigger_type)
                    arguments[event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        a_id_s = argument['argument_start_index']
                        argument_text = argument['argument']
                        a_id_e = a_id_s + len(argument_text)
                        arguments[event_key].append((a_id_s, a_id_e, role))
                    

                self.sent.append([CLS] + words + [SEP])
                self.triggers.append(triggers)
                self.arguments.append(arguments)

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, idx):
        words,triggers, arguments = self.sent[idx], self.triggers[idx], self.arguments[idx]
        tokens_x, is_heads = [], []
        for w in words:
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            tokens_x.extend(tokens_xx), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        token_type_ids = [0] * (len(words))
        for arg in arguments:
            id_s, id_e, trigger_type = arg
            for t in range(id_s, id_e):
                token_type_ids[t+1]=1
                
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)
        seqlen = len(tokens_x)
        mask = [1] * seqlen

        return tokens_x, triggers_y, arguments, seqlen, head_indexes, mask, words, triggers,token_type_ids
            
