from gensim.models import word2vec
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

with open("../input/datas0/data/train_corpus.txt","r") as fin:
    text = fin.read()
with open("../input/datas0/data/test_corpus.txt","r") as fin1:
    text1 = fin1.read()

train_text = text.split()
test_text = text1.split()
text = train_text + test_text
model = word2vec.Word2Vec(text,vector_size=100,hs=1,min_count=1, window=5, workers=4)
model.save('models/word_model')

w2v_path = '/kaggle/working/word_model'
label_dict = {"<START>": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4,"B-ORG": 5,"I-ORG": 6,"O": 7,"<END>": 8}
special_token_list = ["<PADDING>", "<UNKNOWN>"]
class ModelEmbedding:

    def __init__(self, w2v_path):
        # 加载词向量矩阵
        self.word2vector = word2vec.Word2Vec.load(w2v_path).wv
        # 输入的总词数
        self.input_word_list = special_token_list + self.word2vector.index_to_key
        self.embedding = np.concatenate((np.random.rand(2, 100), self.word2vector.vectors))
        self.n_voc = len(self.input_word_list)
        self.n_class = len(label_dict)
        self.get_word2id()
        self.get_tag2id()

    def get_word2id(self):
        # word2id词典
        self.word2id = dict(zip(self.input_word_list, list(range(len(self.input_word_list)))))
        # return self.word2id

    def get_tag2id(self):
        # tag2id词典
        self.tag2id = label_dict


class NERdataset(Dataset):

    def __init__(self, file_dir, word2id, tag2id):
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()
        label = open(label_file).readlines()
        self.corpus = []
        self.label = []
        #self.length = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['UNKNOWN']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            #self.length.append(len(corpus_.split()))
            


        '''self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()'''

    def __getitem__(self, item):
        return torch.Tensor(self.corpus[item]), torch.Tensor(self.label[item])

    def __len__(self):
        return len(self.corpus)
    
def collate_fn(batch):

    corpus_list = [x[0] for x in batch]
    label_list = [x[1] for x in batch]
    lengths = [len(x[0]) for x in batch]
    corpus_list = pad_sequence(corpus_list, padding_value=0)
    label_list = pad_sequence(label_list, padding_value=-1)
    return corpus_list.transpose(0, 1), label_list.transpose(0, 1), lengths


def get_mask(length):
    max_len = int(max(length))
    mask = torch.Tensor()
    # 与每个序列等长度的全1向量连接长度为最大序列长度-当前序列长度的全0向量。
    for len_ in length:
        mask = torch.cat((mask, torch.Tensor([[1] * len_ + [0] * (max_len-len_)])), dim=0)
    return mask


class LstmCrf(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_class,n_voc, model_embedding):
        super(LstmCrf, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.n_voc = n_voc
        # embedding
        self.model_embedding = model_embedding
        self.embedding = nn.Embedding(num_embeddings=self.n_voc,embedding_dim=self.input_dim, padding_idx=0)
        
        self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*hidden_dim, n_class)
        # transition matrix logP(y_i, y_i+1)
        self.transition_matrix = nn.Parameter(torch.rand(n_class, n_class))
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=1)
        
    def reset_parameters(self):
        # 转移概率位于log空间
        init.normal_(self.transition_matrix)
        # score得分为0
        self.transition_matrix.detach()[label_dict["<START>"],:] = -10000
        self.transition_matrix.detach()[:,label_dict["<END>"]] = -10000

    def forward(self, input_data,input_len):

        embedded_input = self.embedding(input_data)
        #input_len = torch.Tensor(input_len).cuda()
        packed_embedded_input = pack_padded_sequence(input=embedded_input, lengths=input_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_embedded_input)
        output, _ = pad_packed_sequence(packed_output)
        output = self.linear(output)
        output = output.transpose(0, 1)
        return output
    
    def forward_alpha(self, emission, mask):
        # 发射矩阵， emission
        # batch大小，最大序列长度
        batch_size, seq_len = mask.size()
        # logα_0
        log_alpha_init = torch.full((batch_size, self.n_class), fill_value=-10000).cuda()
        # i=0时，label为start的概率为1，其余为0,取log
        log_alpha_init[:, 0] = 0
        # alpha, [batch_size, n_class]
        log_alpha = log_alpha_init
        # 该部分一直到计算α_1到α_n
        for w in range(0, seq_len):
            # 取出当前时刻的mask,每个batch中的元素，对应一个n_class*n_class的矩阵，矩阵所有元素相同，与mask[:.w]同
            # [batch_size, 1]
            mask_t = mask[:, w].unsqueeze(-1)
            # [batch_size, n_class]
            current = emission[:, w, :]
            # [batch_size, n_class, n_class]
            log_alpha_matrix = log_alpha.unsqueeze(2).expand(-1, -1, self.n_class)  # 增列
            # [batch_size, n_class, n_class]
            current_matrix = current.unsqueeze(1).expand(-1, self.n_class, -1)  # 增行
            # [batch, n_class, n_class]
            add_matrix = log_alpha_matrix + current_matrix + self.transition_matrix
            log_alpha = torch.logsumexp(add_matrix, dim=1) * mask_t + log_alpha * (1-mask_t)
        # n->END
        log_alpha = log_alpha + self.transition_matrix[:,label_dict["<END>"]].unsqueeze(0)
        total_score = torch.logsumexp(log_alpha, dim=1)
        return total_score
    
    def get_sentence_score(self, emission, labels, mask):
        batch_size, seq_len, n_class = emission.size()
        # 增加<START>label
        # [batch_size, word_vectors]
        labels = torch.cat([labels.new_full((batch_size, 1), fill_value=label_dict["<START>"]), labels], 1)
        # [batch_size, vectors]
        scores = emission.new_zeros(batch_size)

        # M_1到M_n
        for i in range(seq_len):
            # [batch_size, 1]
            mask_i = mask[:, i]
            # [batch_size, n_class]
            current = emission[:, i, :]
            # [batch_size,1]
            emit_score = torch.cat([each_score[next_label].unsqueeze(-1) for each_score, next_label in zip(current, labels[:, i+1])], dim=0)
            # [batch_size,1]
            transition_score = torch.stack([self.transition_matrix[labels[b, i],labels[b, i+1]] for b in range(batch_size)])
            scores += (emit_score + transition_score) * mask_i
        transition_to_end = torch.stack([self.transition_matrix[label[mask[b,:].sum().long()], label_dict["<END>"]] for b, label in enumerate(labels)])
        scores += transition_to_end
        return scores
    
    def get_loss(self, emission, labels, mask):
        # log_Z
        log_Z = self.forward_alpha(emission, mask)
        # log_y
        log_alpha_n = self.get_sentence_score(emission, labels, mask)
        # -log(y/z)
        loss = torch.mean(log_Z - log_alpha_n)
        return loss
    
    def get_best_path(self, emission, mask):
        # 发射矩阵， emission
        # batch大小，最大序列长度
        batch_size, seq_len = mask.size()
        log_alpha_init = torch.full((batch_size, self.n_class), fill_value=-10000).cuda()
        log_alpha_init[:, 0] = 0
        log_alpha = log_alpha_init
        #print('emission:',emission)
        #print('log_alpha:',log_alpha[0])
        #print('mask:',mask)
        # 指针
        pointers = []
        for w in range(0, seq_len):
            # [batch_size, 1]
            mask_t = mask[:, w].unsqueeze(-1)
            # [batch_size, n_class]，当前的词对应的emission
            current = emission[:, w, :]
            log_alpha_matrix = log_alpha.unsqueeze(2).expand(-1, -1, self.n_class)  # 增列
            trans = log_alpha_matrix + self.transition_matrix
            # 选取上一时刻的y_i-1使得到当前时刻的某个y_i的路径分数最大,max_trans[batch_size, n_class]
            max_trans, pointer = torch.max(trans, dim=1)
            # 添加路径,注意此时的pointer指向的是上一时刻的label
            pointers.append(pointer)
            # 获取当前时刻新的log_alpha [batch_size, n_class]
            cur_log_alpha = current + max_trans
            # 根据mask判断是否更新，得到当前时刻的log_alpha
            log_alpha = log_alpha * (1-mask_t) + cur_log_alpha * mask_t
            #print(log_alpha[0])
            #if w == 1:
                #print(log_alpha_matrix[0])
                #print(trans[0])
                #print(max_trans[0], pointer[0])
        # 将pointers转为张量, [batch_size, seq_len, n_class]
        pointers = torch.stack(pointers, 1)
        # n->END
        log_alpha = log_alpha + self.transition_matrix[label_dict["<END>"]]
        # 找到n->END的最优路径, [batch_size], [batch_size]
        best_log_alpha, best_label = torch.max(log_alpha, dim=1)
        best_path = []
        # 从后向前，不断寻路(注意不同数据的路长不同，不同数据单独寻路)
        for i in range(batch_size):
            # 当前数据的路径长度
            seq_len_i = int(mask[i].sum())
            # 当前数据对应的有效pointers[seq_len_i, n_class]
            pointers_i = pointers[i, :seq_len_i]
            # 当前数据的best_label
            best_label_i = best_label[i]
            # 遍历寻路,当前数据的路径
            best_path_i = [best_label_i]
            for j in range(seq_len_i):
                # 从后向前遍历
                index = seq_len_i-j-1
                # 当前时刻的best_label_i
                best_label_i = pointers_i[index][best_label_i]
                best_path_i = [best_label_i] + best_path_i
            # 除去时刻1之前的的路径
            best_path_i = best_path_i[1:]
            # 添加到总路径中
            best_path.append(best_path_i)
        return best_path

# embedding
w2v_path = '/kaggle/working/word_model'
model_embedding = ModelEmbedding(w2v_path)
# model建立
model = LstmCrf(input_dim=100, hidden_dim=100, n_class=model_embedding.n_class,n_voc=model_embedding.n_voc, model_embedding=model_embedding).cuda()
#model = torch.load('/kaggle/working/bilstm_crf.pth')
# 优化器
optim = torch.optim.Adam(model.parameters(), lr=0.02)
# 得到word2id与tag2id字典，构造dataset
word2id = model_embedding.word2id
tag2id = model_embedding.tag2id
#vectors = model_embedding.embedding
# 训练集，测试集
train_dataset = NERdataset('../input/datas0/data/train',word2id,tag2id)
test_dataset = NERdataset('../input/datas0/data/test',word2id,tag2id)
# 构造dataloader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, collate_fn=collate_fn,shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, collate_fn=collate_fn)
for epoch in range(30):
    print('epoch: {}'.format(epoch+1))
    sum_loss = 0
    num = 0
    for i,data in enumerate(train_dataloader):
        optim.zero_grad()
        x,y,lengths = data
        x = x.cuda().long()
        y = y.cuda().long()
        mask = get_mask(lengths)
        mask = torch.Tensor(mask).cuda()
        emission = model.forward(input_data=x,input_len=lengths)
        loss = model.get_loss(emission=emission, labels=y, mask=mask)
        if (i % 20 == 0):
                print('epoch: ', epoch+1, ' step:%04d,------------loss:%f' % (i, loss.item()))
        sum_loss += loss.item()
        num += 1
        loss.backward()
        optim.step()
    print('batch loss:{}'.format(sum_loss/num))
    torch.save(model,'/kaggle/working/bilstm-crf—256.pth')

w2v_path = '/kaggle/working/word_model'
model_embedding = ModelEmbedding(w2v_path)
word2id = model_embedding.word2id
tag2id = model_embedding.tag2id
test_dataset = NERdataset('../input/datas0/data/test',word2id,tag2id)
model = LstmCrf(input_dim=100, hidden_dim=100, n_class=model_embedding.n_class,n_voc=model_embedding.n_voc, model_embedding=model_embedding).cuda()
model = torch.load('../input/models/bilstm-crf-256-0.79.pth')
model.eval()
preds, labels = [], []
test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, collate_fn=collate_fn)
train_dataset = NERdataset('../input/datas0/data/train',word2id,tag2id)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, collate_fn=collate_fn)
for index, data in enumerate(test_dataloader):
    #optim.zero_grad()
    corpus, label, length = data
    corpus = corpus.cuda().long()
    mask = get_mask(length)
    mask = torch.Tensor(mask).cuda()
    emission = model.forward(input_data=corpus,input_len=length)
    label = label.tolist()
    for i in label:
        for j in i:
            if j != -1:
                labels.append(j)
    path = model.get_best_path(emission,mask)
    for var in path:
        var = torch.Tensor(var).detach().cpu().tolist()
        for tep in var:
            preds.append(tep)
        
precision = precision_score(labels, preds, average='macro')
recall = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')
report = classification_report(labels, preds)
print(report)