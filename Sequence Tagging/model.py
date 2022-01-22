import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
label_dict = {"<START>": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4,"B-ORG": 5,"I-ORG": 6,"O": 7,"<END>": 8}

class LstmCrf(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_class,n_voc, model_embedding,label_dict):
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