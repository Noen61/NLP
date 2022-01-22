import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence

from utils import ModelEmbedding,collate_fn,data_processing,get_mask
from dataset import NERdataset
from model import LstmCrf

if __name__ == '__main__':
    label_dict = {"<START>": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4,"B-ORG": 5,"I-ORG": 6,"O": 7,"<END>": 8}
    # embedding
    w2v_path = 'models/word_model'
    model_embedding = ModelEmbedding(w2v_path,label_dict)
    # model建立
    model = LstmCrf(input_dim=100, hidden_dim=100, n_class=model_embedding.n_class,
                    n_voc=model_embedding.n_voc, model_embedding=model_embedding).cuda()
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=0.02)
    # 得到word2id与tag2id字典，构造dataset
    word2id = model_embedding.word2id
    tag2id = model_embedding.tag2id
    # 训练集，测试集
    train_dataset = NERdataset('data/train',word2id,tag2id)
    test_dataset = NERdataset('data/test',word2id,tag2id)
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
        torch.save(model,'models/bilstm-crf.pth')