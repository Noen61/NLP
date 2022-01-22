import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader

class NERdataset(Dataset):

    def __init__(self, file_dir, word2id, tag2id):
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()
        label = open(label_file).readlines()
        self.corpus = []
        self.label = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['UNKNOWN']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            

    def __getitem__(self, item):
        return torch.Tensor(self.corpus[item]), torch.Tensor(self.label[item])

    def __len__(self):
        return len(self.corpus)