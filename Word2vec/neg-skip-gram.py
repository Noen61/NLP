import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import Counter 
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from string import punctuation
import re
import sys

MAX_VOCAB_SIZE = 30000
C= 3 #周围单词个数（context window）
K = 100 #下采样（number of negative samoles）

punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
#with open("../input/textdata/data/zh.txt","r") as fin:
with open("../input/textdata/data/en.txt","r") as fin:
    text = fin.read()
text = re.sub(r"[{}]+".format(punc)," ",text)
text = text.split()
#构造词频字典
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
#未知词
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
#建立词和索引的对应
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}
#计算词频,按照原论文转换为3/4次方
word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs = word_counts/np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word) 

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

#构造神经网络，输入中心词，背景词，负样本词，输出词向量 
class EmbeddingModel(nn.Module):
    def __init__(self,vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()    
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        #权重初始化
        initrange = 0.5 / self.embed_size
        #中心词向量表
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        #背景词向量表
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)   
    def forward(self, input_labels, pos_labels, neg_labels): 
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels) # [batch_size, (windows_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels).neg() # [batch_size, (windows_size * 2 * K), embed_size]
        #权重计算
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]新增一个维度用于向量乘法
        pos_dot = torch.bmm(pos_embedding,input_embedding).squeeze(2) # [batch_size, (windows_size * 2)]
        neg_dot = torch.bmm(neg_embedding,input_embedding).squeeze(2) # [batch_size, (windows_size * 2 * K)]
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = -(log_pos + log_neg)   
        return loss 
    def input_embeddings(self):
        ##取出词向量
        return self.in_embed.weight.data.cpu().numpy()

NUM_EPOCHS =  10 #迭代次数
BATCH_SIZE = 512 #批样本数
LEARNING_RATE = 0.02 #学习率
EMBEDDING_SIZE = 100 #词向量长度
save_path = r'/kaggle/working/'
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
for epoch in range(NUM_EPOCHS):
    sum = 0
    j = 0
    for i, (input_labels, pos_labels,neg_labels) in enumerate(dataloader):

        input_labels = input_labels.long().to(device) 
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)
        
        optimizer.zero_grad() 
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()
        sum += loss.item()
        j += 1
    torch.save(model.state_dict(), save_path+"embedding-{}.th".format(EMBEDDING_SIZE))
    #torch.save(model.state_dict(), save_path+"cn-embedding-{}.th".format(EMBEDDING_SIZE))
    print("epoch", epoch, "loss", sum/j)

#获取词向量并将其写入csv文件中
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
model.load_state_dict(torch.load(save_path+"embedding-{}.th".format(EMBEDDING_SIZE),map_location='cpu'))
#model.load_state_dict(torch.load(save_path+"cn-embedding-{}.th".format(EMBEDDING_SIZE),map_location='cpu'))
X = model.input_embeddings()
col = ['word']
for i in range(len(X[0])):
    col.append(str(i+1))
temp = np.array(idx_to_word).reshape(len(idx_to_word),1)
Y = np.c_[temp,X]
pd_data = pd.DataFrame(Y,columns=col)
pd_data.to_csv('/kaggle/working/en-vectors.csv')
#pd_data.to_csv('/kaggle/working/cn-vectors.csv')