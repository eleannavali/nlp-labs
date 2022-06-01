import torch
from torch import nn
from config import  EMB_DIM, MAX_LENGTH

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=True) :
        """

        Args:
            output_size(int): the number of classes
            embeddings(nparray):  the  2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__() 

        # 1 - define the embedding layer
        self.embed_l = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=trainable_emb)  # EX4
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned

        # 4 - define a non-linear transformation of the representations
        self.linear1 = nn.Linear(EMB_DIM, 60)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(60, output_size) # EX5
        

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        ...  # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # print("Type:",type(x), type(x[0]))

        # 1 - embed the words, using the embedding layer
        embeddings = self.embed_l(x) # EX6  batch, 40, 50
    
        #TODO: we have to mask padding zeros before mean()
        # 2 - construct a sentence representation out of the word embeddings
        #representations =  torch.mean(embeddings, dim=1) # EX6  batch, 50

        # sum and then divide by the real length of the sentence. 
        representations = torch.sum(embeddings, dim=1)
        for i in range(lengths.shape[0]) :
            representations[i] = representations[i] / lengths[i]
        

        # 3 - transform the representations to new ones.
        representations = self.linear2(representations)
        representations = self.relu(representations) # EX6
        
        # 4 - project the representations to classes using a linear layer
        logits = self.linear1(representations)  # EX6 batch, class_size

        return logits


# LSTM CLASS

class LSTM(nn.Module) :
    def __init__(self, output_size, embeddings) -> None:
        super(LSTM, self).__init__()

        self.hidden_size = 120
        self.num_layers = 1 
        self.representation_size = self.hidden_size
        self.output_size = output_size

        # Layers
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False)
        num_embeddings, emb_dim = embeddings.shape
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(self.representation_size, self.output_size)

    def forward(self, x, lengths) :
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)

        # Helps the lstm ignore the padded zeros
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)
        # Sentence representation as the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_size).float()
        for i in range(lengths.shape[ 0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]

        logits = self.linear(representations)
        return logits
