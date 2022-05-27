import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# print(word2idx)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to 
lab_encoder = LabelEncoder()
lab_encoder.fit(y_train)
y_train = lab_encoder.transform(y_train) #EX1
y_test = lab_encoder.transform(y_test)   #EX1
n_classes = lab_encoder.classes_  # EX1 - LabelEncoder.classes_.size
print('Total number of classes ', n_classes)
print("First 10 labels : ", y_train[0:10])
print("Coressponding to : ", lab_encoder.inverse_transform(y_train[0:10]))

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# print("dataset tuple:",train_set[0])

# # EX4 - Define our PyTorch-based DataLoader
# train_loader = ...  # EX7
# test_loader = ...  # EX7

# #############################################################################
# # Model Definition (Model, Loss Function, Optimizer)
# #############################################################################
# model = BaselineDNN(output_size=...,  # EX8
#                     embeddings=embeddings,
#                     trainable_emb=EMB_TRAINABLE)

# # move the mode weight to cpu or gpu
# model.to(DEVICE)
# print(model)

# # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
# criterion = ...  # EX8
# parameters = ...  # EX8
# optimizer = ...  # EX8

# #############################################################################
# # Training Pipeline
# #############################################################################
# for epoch in range(1, EPOCHS + 1):
#     # train the model for one epoch
#     train_dataset(epoch, train_loader, model, criterion, optimizer)

#     # evaluate the performance of the model, on both data sets
#     train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
#                                                             model,
#                                                             criterion)

#     test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
#                                                          model,
#                                                          criterion)
