from gensim.models import word2vec
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence

class ModelEmbedding:

    def __init__(self, w2v_path,label_dict):
        # 加载词向量矩阵
        self.word2vector = word2vec.Word2Vec.load(w2v_path).wv
        # 输入的总词数
        self.input_word_list = special_token_list + self.word2vector.index_to_key
        self.embedding = np.concatenate((np.random.rand(2, 100), self.word2vector.vectors))
        self.n_voc = len(self.input_word_list)
        self.n_class = len(label_dict)
        self.get_word2id()
        #tag2id词典
        self.tag2id = label_dict

    def get_word2id(self):
        # word2id词典
        self.word2id = dict(zip(self.input_word_list, list(range(len(self.input_word_list)))))
        # return self.word2id

    
def collate_fn(batch):

    corpus_list = [x[0] for x in batch]
    label_list = [x[1] for x in batch]
    lengths = [len(x[0]) for x in batch]
    corpus_list = pad_sequence(corpus_list, padding_value=0)
    label_list = pad_sequence(label_list, padding_value=-1)
    return corpus_list.transpose(0, 1), label_list.transpose(0, 1), lengths

def data_processing(train_path,test_path):
    with open(train_path,"r") as train_fin:
        train_text = train_fin.read()
    with open(test_path,"r") as test_fin:
        test_text = test_fin.read()

    train_text = train_text.split()
    test_text = test_text.split()
    text = train_text + test_text
    special_token_list = ["<PADDING>", "<UNKNOWN>"]
    vocab = dict(Counter(text))
    vocab["<UNKNOWN>"] = 1
    vocab["<PADDING>"] = 1
    id2word = [word for word in vocab.keys()]
    word2id = {word:i for i,word in enumerate(id2word)}
    tag2id = {"<START>": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4,"B-ORG": 5,"I-ORG": 6,"O": 7,"<END>": 8}
    return word2id,tag2id

def get_mask(length):
    max_len = int(max(length))
    mask = torch.Tensor()
    # 与每个序列等长度的全1向量连接长度为最大序列长度-当前序列长度的全0向量。
    for len_ in length:
        mask = torch.cat((mask, torch.Tensor([[1] * len_ + [0] * (max_len-len_)])), dim=0)
    return mask