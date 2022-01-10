import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data



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