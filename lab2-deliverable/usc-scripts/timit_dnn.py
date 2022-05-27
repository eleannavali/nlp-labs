# Train a torch DNN for Kaldi DNN-HMM model

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_dataset import TorchSpeechDataset
from torch_dnn import TorchDNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 4
HIDDEN_DIM = [256,256,128,64]
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 10
PATIENCE = 3

# if len(sys.argv) < 2:
#     print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/checkpoint/weights2.pt'


# FIXME: You may need to change these paths
TRAIN_ALIGNMENT_DIR = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_train_ali'
DEV_ALIGNMENT_DIR = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_dev_ali'
TEST_ALIGNMENT_DIR = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_test_ali'


def train(model, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    """Train model using Early Stopping and save the checkpoint for
    the best validation loss
    """
    
    plot_loss_tr=[]
    plot_loss_dev=[]
    plot_ac_tr=[]
    plot_ac_dev=[]
    for epoch in range(epochs):
        run_loss=0
        dev_loss=0
        cor_dev=0
        cor_tr=0
        model.train()
        for i,data in enumerate(train_loader):
            X_batch,y_batch = data
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out,y_batch)
            loss.backward()
            optimizer.step()

            run_loss += loss.detach().item()
            cor_tr+=(torch.argmax(out,dim=1)==y_batch).float().sum()

        plot_loss_tr.append(run_loss/(i+1))
        plot_ac_tr.append(100*cor_tr/(i*128))

        model.eval()
        for i,data in enumerate(dev_loader):
            X_batch,y_batch = data
            out = model(X_batch)
            loss = criterion(out,y_batch)

            dev_loss += loss.detach().item()
            # print('------------ ',torch.argmax(out,dim=1)[0],y_batch[0],(torch.argmax(out,dim=1)==y_batch).float().sum())
            cor_dev+=(torch.argmax(out,dim=1).item()==y_batch).float().sum()

        plot_loss_dev.append(dev_loss/(i+1))
        plot_ac_dev.append(100*cor_dev/(i*128))

        print("*************",i,sum(1 for _ in train_loader))
        print("For epoch ",epoch," train accuracy=",100*cor_tr/(i*128)," and dev accuracy=",100*cor_dev/(i*128))

        if epoch % 5 == 0: 
            print("Epoch: %d, train loss: %1.5f, dev loss: %1.5f", epoch, plot_loss_tr[epoch],plot_loss_dev[epoch])
    
    fig , ax = plt.subplots(figsize = (10,10))

    plt.plot(plot_loss_dev, label='Val_loss') #actual plot
    plt.plot(plot_loss_tr, label='Train_loss') #predicted plot
    plt.title('Training and validation loss curves')
    plt.legend()
    plt.show()

    fig , ax = plt.subplots(figsize = (10,10))

    plt.plot(plot_ac_dev, label='Val_accuracy') #actual plot
    plt.plot(plot_ac_tr, label='Train_accuracy') #predicted plot
    plt.title('Training and validation accuracy curves')
    plt.legend()
    plt.show()
    


trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')

scaler = StandardScaler()
scaler.fit(trainset.feats)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)


dnn = TorchDNN(
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P
)
dnn.to(DEVICE)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
dev_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(dnn.parameters(),lr=0.003)
criterion = nn.CrossEntropyLoss()

train(dnn, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
torch.save(dnn,BEST_CHECKPOINT)
