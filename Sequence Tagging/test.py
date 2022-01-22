import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence

from utils import ModelEmbedding,collate_fn,data_processing,get_mask
from dataset import NERdataset
from model import LstmCrf
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

if __name__ == '__main__':
    label_dict = {"<START>": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4,"B-ORG": 5,"I-ORG": 6,"O": 7,"<END>": 8}
    w2v_path = 'models/word_model'
    model_embedding = ModelEmbedding(w2v_path,label_dict)
    word2id = model_embedding.word2id
    tag2id = model_embedding.tag2id
    test_dataset = NERdataset('data/test',word2id,tag2id)
    model = LstmCrf(input_dim=100, hidden_dim=100, n_class=model_embedding.n_class,n_voc=model_embedding.n_voc, model_embedding=model_embedding).cuda()
    model = torch.load('models/bilstm-crf-256-0.79.pth')
    model.eval()
    preds, labels = [], []
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, collate_fn=collate_fn)
    for index, data in enumerate(test_dataloader):
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
    print(len(preds))