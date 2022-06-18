from errno import ELIBBAD
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from models import Attention_DNN, LSTM_Attention
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

for DATASET in ["Semeval2017A"]: # "MR",
    for MODEL in ["DNNAttention", "LSTMAttention"]:

        PATH = './best_models/'+DATASET+'_'+MODEL+'_model.pt'
        if MODEL == "MR":
            if MODEL == "DNNAttention":
                model = Attention_DNN(output_size=2,embeddings=embeddings)
            else:
                model = LSTM_Attention(output_size=2,embeddings=embeddings,bidirectional=True)
        else:
            if MODEL == "DNNAttention":
                model = Attention_DNN(output_size=3,embeddings=embeddings)
            else:
                model = LSTM_Attention(output_size=3,embeddings=embeddings,bidirectional=True)

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
        
        # convert data labels from strings to 
        lab_encoder = LabelEncoder()
        lab_encoder.fit(y_train)
        y_train = lab_encoder.transform(y_train) #EX1
        y_test = lab_encoder.transform(y_test)   #EX1
        n_classes = lab_encoder.classes_

        if DATASET == "MR":
            # Torch one-hot
            y_train = torch.Tensor.numpy(torch.nn.functional.one_hot(torch.tensor(y_train,dtype=torch.long),len(n_classes))*1.)
            y_test = torch.Tensor.numpy(torch.nn.functional.one_hot(torch.tensor(y_test,dtype=torch.long),len(n_classes))*1.)


        test_set = SentenceDataset(X_test, y_test, word2idx)
        test_loader =  DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        model.eval()

        # obtain the model's device ID
        device = next(model.parameters()).device
        with open('./results/visualization/data_'+DATASET+'_'+MODEL+'.json', "a") as f:
            f.write('[\n')
            # f.close()
        sampleid = 0
        # IMPORTANT: in evaluation mode, we don't want to keep the gradients
        # so we do everything under torch.no_grad()
        with torch.no_grad():
            for index, batch in enumerate(test_loader, 1):
                # get the inputs (batch)
                inputs, labels, lengths = batch
                # print(inputs[0])
                # for i in inputs[0]:
                #     print(i.item(),idx2word[i.item()])

                # Step 1 - move the batch tensors to the right device
                inputs.to(device) # EX9
                labels.to(device)
                lengths.to(device)  

                # Step 2 - forward pass: y' = model(x)
                pred, scores = model(inputs, lengths) # EX9  # EX9

                # Step 4 - make predictions (class = argmax of posteriors)
                class_pred = torch.argmax(pred, dim=1)  # EX9
                for j,sample in enumerate(inputs):
                    with open('./results/visualization/data_'+DATASET+'_'+MODEL+'.json', "a") as f:       
                        f.write('{\n')
                        f.write('"text": [\n')
                        f.write('"'+idx2word[sample[0].item()]+'"')
                        for i in sample[1:]:
                            if (i.item() == 0):
                                break
                            f.write(',\n"'+idx2word[i.item()]+'"')
                        f.write('\n],\n')
                        if DATASET == "MR":
                            f.write('"label": '+ str(torch.argmax(labels[j]).item())+",\n")
                        else:
                            f.write('"label": '+ str(labels[j].item())+",\n")
                        f.write('"prediction": '+ str(class_pred[j].item())+",\n")
                        f.write('"attention": [\n')
                        for score in scores[j][:-1]:
                            f.write(str(score.item())+",\n")
                        f.write(str(scores[j][-1].item())+"\n],\n")
                        f.write('"id": "sample_'+str(sampleid)+'"\n')
                        f.write('},\n')
                        # f.close()

                        sampleid+=1

            with open('./results/visualization/data_'+DATASET+'_'+MODEL+'.json', "a") as f:
                f.write(']')
                f.close()
