import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN,LSTM,Attention_DNN, LSTM_Attention
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from utils.plotting import plot_training_curves
import numpy as np


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
EMB_TRAINABLE = True
BATCH_SIZE = 64

EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"
MODEL = "LSTMAttention" # options: "DNN", "LSTM", "DNNAttention", "LSTMAttention"
CONCAT = True
BIDIRECTIONAL = True

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# MODE : Train all models 
for DATASET in [ "MR","Semeval2017A"]: 
    for MODEL in ["DNN", "LSTM", "DNNAttention", "LSTMAttention"]:
        for CONCAT in [True,False]:
            if MODEL=="DNNAttention" or MODEL=="LSTMAttention":
                if CONCAT==False:
                    break
                else:
                    CONCAT='None'
            for BIDIRECTIONAL in [True,False]:
                if MODEL=="DNNAttention" or MODEL=="DNN":
                    if BIDIRECTIONAL==False:
                        break
                    else:
                        BIDIRECTIONAL='None'

    ## MODE : Save the best attention model for each dataset
    # for MODEL in ["DNNAttention", "LSTMAttention"]:
    #             if DATASET=='MR':
    #                 if MODEL=='DNNAttention':
    #                     EPOCHS=47
    #                     CONCAT = 'None'
    #                     BIDIRECTIONAL = 'None'
    #                 else:
    #                     EPOCHS=45
    #                     CONCAT = 'None'
    #                     BIDIRECTIONAL = True
    #             else:
    #                 if MODEL=='DNNAttention':
    #                     EPOCHS=43
    #                     CONCAT = 'None'
    #                     BIDIRECTIONAL = 'None'
    #                 else:
    #                     EPOCHS=42
    #                     CONCAT = "None"
    #                     BIDIRECTIONAL = True

            

                ## main code ...
                print("> "+DATASET+" "+MODEL+" "+str(CONCAT)+" "+str(BIDIRECTIONAL))

                # print(word2idx)

                # load the raw data
                if DATASET == "Semeval2017A":
                    X_train, y_train, X_test, y_test = load_Semeval2017A()
                    # one_hot_encoder = {0:np.array([1., 0., 0.]), 1:np.array([0., 1., 0.]), 2:np.array([0., 0., 1.])}
                elif DATASET == "MR":
                    X_train, y_train, X_test, y_test = load_MR()
                    # one_hot_encoder = {0:np.array([1., 0.]), 1:np.array([0., 1.])}
                else:
                    raise ValueError("Invalid dataset")

                # convert data labels from strings to 
                lab_encoder = LabelEncoder()
                lab_encoder.fit(y_train)
                y_train = lab_encoder.transform(y_train) #EX1
                y_test = lab_encoder.transform(y_test)   #EX1
                n_classes = lab_encoder.classes_  # EX1 - LabelEncoder.classes_.size

                # print('Total number of classes ', len(n_classes))
                # print("First 10 labels : ", y_train[0:10])
                # print("Coressponding to : ", lab_encoder.inverse_transform(y_train[0:10]))

                # BCEWithLogitsLoss needs one-hot encoded labels vs CrossEntropyLoss needs integers as labels
                if DATASET == "MR":
                    # Custom one-hot
                    # y_train = [one_hot_encoder[y] for y in y_train]
                    # y_test = [one_hot_encoder[y] for y in y_test]

                    # Torch one-hot
                    y_train = torch.Tensor.numpy(torch.nn.functional.one_hot(torch.tensor(y_train,dtype=torch.long),len(n_classes))*1.)
                    y_test = torch.Tensor.numpy(torch.nn.functional.one_hot(torch.tensor(y_test,dtype=torch.long),len(n_classes))*1.)


                # Define our PyTorch-based Dataset
                train_set = SentenceDataset(X_train, y_train, word2idx)
                test_set = SentenceDataset(X_test, y_test, word2idx)

                # print("dataset tuple:",train_set[0])

                # # EX4 - Define our PyTorch-based DataLoader
                train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)    # EX7
                test_loader =  DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) # EX7

                #############################################################################
                # Model Definition (Model, Loss Function, Optimizer)
                #############################################################################
                if DATASET=='MR':
                    if MODEL=="DNN":
                        model = BaselineDNN(output_size=2,  # EX8
                                            embeddings=embeddings,
                                            trainable_emb=EMB_TRAINABLE,concat=CONCAT)
                    elif MODEL=="LSTM":
                        model = LSTM(output_size=2,embeddings=embeddings,concat=CONCAT,bidirectional=BIDIRECTIONAL)
                    elif MODEL=="DNNAttention":
                        model = Attention_DNN(output_size=2,embeddings=embeddings)
                    else:
                        model = LSTM_Attention(output_size=2,embeddings=embeddings,bidirectional=BIDIRECTIONAL)
                else:
                    if MODEL=="DNN":
                        model = BaselineDNN(output_size=3,  # EX8
                                            embeddings=embeddings,
                                            trainable_emb=EMB_TRAINABLE)
                    elif MODEL=="LSTM":
                        model = LSTM(output_size=3,embeddings=embeddings,concat=CONCAT,bidirectional=BIDIRECTIONAL)
                    elif MODEL=="DNNAttention":
                        model = Attention_DNN(output_size=3,embeddings=embeddings)
                    else:
                        model = LSTM_Attention(output_size=3,embeddings=embeddings,bidirectional=BIDIRECTIONAL)

                # # move the mode weight to cpu or gpu
                model.to(DEVICE)
                print(model)

                # for x,y,k in train_loader:
                #     pred = model(x,k)
                #     print(pred[0],y[0])
                #     exit(0)

                # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
                if DATASET=='MR':
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                parameters =  (p for p in model.parameters() if p.requires_grad)
                optimizer = torch.optim.Adam(parameters, lr=1e-4)

                #############################################################################
                # Training Pipeline
                #############################################################################
                tr_loss = []
                val_loss = []
                tr_acc = []
                val_acc =[]
                tr_f1 = []
                val_f1 = []
                tr_rec = []
                val_rec = []
                min_loss = 10000
                for epoch in range(1, EPOCHS + 1):
                    # train the model for one epoch
                    train_dataset(epoch, train_loader, model, criterion, optimizer)

                    # evaluate the performance of the model, on both data sets
                    if DATASET=='MR':
                        train_loss, (y_pred_train, y_train), accuracy_train, f1_train, recall_train = eval_dataset(train_loader,model,criterion)

                        test_loss,(y_pred_test, y_test),  accuracy_test, f1_test, recall_test = eval_dataset(test_loader,model,criterion)
                    else:
                        train_loss, (y_pred_train, y_train), accuracy_train, f1_train, recall_train = eval_dataset(train_loader,model,criterion,binary_classification=False)

                        test_loss,(y_pred_test, y_test),  accuracy_test, f1_test, recall_test = eval_dataset(test_loader,model,criterion,binary_classification=False)
                    
                    with open('./results/out_'+DATASET+'_'+MODEL+"_batch_size="+str(BATCH_SIZE)+"_epochs="+str(EPOCHS)+"_concat="+str(CONCAT)+"_bidirectional="+str(BIDIRECTIONAL)+'.txt', "a") as f:
                        f.write("For epoch " +str(epoch) +":\n")
                        f.write("F1 score for TRAINING: "+str(f1_train)+"\n")
                        f.write("F1 score for TEST: "+str(f1_test)+"\n")
                        f.write("Recall for TRAINING: "+ str(recall_train)+"\n")
                        f.write("Recall for TEST: "+ str(recall_test)+"\n")
                        f.close()
                    
                    tr_loss.append(train_loss)
                    val_loss.append(test_loss)
                    tr_acc.append(accuracy_train)
                    val_acc.append(accuracy_test)
                    tr_f1.append(f1_train)
                    val_f1.append(f1_test)
                    tr_rec.append(recall_train)
                    val_rec.append(recall_test)

                plot_training_curves(tr_loss,tr_acc,val_loss,val_acc,DATASET,MODEL,BATCH_SIZE, EPOCHS,CONCAT,BIDIRECTIONAL)
                print("acc total:",tr_acc)

                with open('./results/best_out_'+DATASET+'_'+MODEL+"_batch_size="+str(BATCH_SIZE)+"_epochs="+str(EPOCHS)+"_concat="+str(CONCAT)+"_bidirectional="+str(BIDIRECTIONAL)+'.txt', "w") as f:
                    f.write("Best f1 score="+str(max(tr_f1))+" for epoch "+str(np.argmax(tr_f1))+" in training set.\n")
                    f.write("Best f1 score="+str(max(val_f1))+" for epoch "+str(np.argmax(val_f1))+" in test set.\n")

                    f.write("Best recall score="+str(max(tr_rec))+" for epoch "+str(np.argmax(tr_rec))+" in training set.\n")
                    f.write("Best recall score="+str(max(val_rec))+" for epoch "+str(np.argmax(val_rec))+" in test set.\n")
                    f.close()

            
                    # We chose the best model for each dataset from the diagrams of the training phase
                    if min_loss>test_loss:
                        min_loss=test_loss
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': test_loss,
                        }, './best_models/'+DATASET+'_'+MODEL+'_model.pt')