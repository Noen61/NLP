import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from const import NUM_EPOCHS,BATCH_SIZE,LEARNING_RATE,EMBEDDING_SIZE,C,K
from utils import build_vocab
from dataset import  WordEmbeddingDataset
from model import EmbeddingModel

def train(model, iterator, optimizer):
    model.train()
    total_loss = 0
    num = 0
    for i, (input_labels, pos_labels,neg_labels) in enumerate(iterator):

        input_labels = input_labels.long().to(device) 
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)
        
        optimizer.zero_grad() 
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num  += 1
    print("loss: {}".format(total_loss/(num)))
            

if __name__ == '__main__':
    train_path = 'data/en.txt'
    idx_to_word,word_to_idx,word_freqs,VOCAB_SIZE = build_vocab(train_path)
    train_dataset = WordEmbeddingDataset(word_to_idx, idx_to_word, word_freqs, word_counts)
    train_loader = data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    device=torch.device("cpu")
    if torch.cuda.is_available():
        device = 'cuda'
    model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
            print("=========train at epoch:{}=========".format(epoch))
            train(model,train_loader, optimizer)
            torch.save(model, "models/vectors.pth")