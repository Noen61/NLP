import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from const import C,K


class WordEmbeddingDataset(data.Dataset):
    def __init__(self,text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded = [word_to_idx.get(word,word_to_idx["<unk>"]) for word in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx #词：索引 的键值对
        self.idx_to_word = idx_to_word #词（列表）
        self.word_freqs = torch.Tensor(word_freqs) #词频率
        self.word_counts = torch.Tensor(word_counts) #词个数
        
    def __len__(self):
        return len(self.text_encoded)
        
    def __getitem__(self,idx):
        #根据idx返回
        center_word = self.text_encoded[idx] #中心词
        pos_indices = list(range(idx-C, idx)) + list((idx+1, idx+C+1)) #中心词前后各C个词作为正样本
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] #取余,以防超过文档范围
        pos_words = self.text_encoded[pos_indices] #周围词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0],False)
        
        return center_word, pos_words, neg_words