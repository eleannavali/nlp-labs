import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import torch
from models import LSTM
from torch.utils.data import DataLoader
from config import EMB_PATH
from dataloading import SentenceDataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")
EMB_DIM = 50
BATCH_SIZE = 64

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

for DATASET in ["MR", "Semeval2017A"]:

    PATH = './best_models/'+DATASET+'_model.pt'

    model = LSTM()
    parameters =  (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=1e-4)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.to(DEVICE)
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
        # one_hot_encoder = {0:np.array([1., 0., 0.]), 1:np.array([0., 1., 0.]), 2:np.array([0., 0., 1.])}
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
        # one_hot_encoder = {0:np.array([1., 0.]), 1:np.array([0., 1.])}
    else:
        raise ValueError("Invalid dataset")

    test_set = SentenceDataset(X_test, y_test, word2idx)
    test_loader =  DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    model.eval()
    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            # get the inputs (batch)
            inputs, labels, lengths = batch

            # Step 1 - move the batch tensors to the right device
            inputs.to(device) # EX9
            labels.to(device)
            lengths.to(device)  

            # Step 2 - forward pass: y' = model(x)
            pred = model(inputs, lengths) # EX9  # EX9

            # Step 4 - make predictions (class = argmax of posteriors)
            class_pred = torch.argmax(pred, dim=1)  # EX9

            # Step 5 - collect the predictions, gold labels
            y_pred.append(class_pred) 
            y.append(labels)
