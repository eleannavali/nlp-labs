import torch
from torch import embedding, nn
from config import  EMB_DIM, MAX_LENGTH

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=True, concat=False) :
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
        self.concat = concat 

        # 4 - define a non-linear transformation of the representations
        if self.concat == True :
            self.linear1 = nn.Linear(2*EMB_DIM, 60)
        else :
            self.linear1 = nn.Linear(EMB_DIM, 60)
        self.relu = nn.ReLU()
        # maps the representations to the classes
        self.linear2 = nn.Linear(60, output_size) # EX5
        

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # print("Type:",type(x), type(x[0]))

        # 1 - embed the words, using the embedding layer
        embeddings = self.embed_l(x) # EX6  batch, 40, 50
    

        # Compute mean in axis 1 
        # sum and then divide by the real length of the sentence. 
        representations = torch.sum(embeddings, dim=1)
        for i in range(lengths.shape[0]) :
            representations[i] = representations[i] / lengths[i]

        if self.concat==True :
            representations2,_ = torch.max(embeddings, dim=1)
            representations = torch.cat((representations, representations2), dim=1)
    
            

        # 3 - transform the representations to new ones.
        representations = self.linear1(representations)
        representations = self.relu(representations) # EX6
        
        # 4 - project the representations to classes using a linear layer
        logits = self.linear2(representations)  # EX6 batch, class_size

        return logits


# LSTM CLASS
class LSTM(nn.Module) :
    def __init__(self, output_size, embeddings, concat=False) -> None:
        super(LSTM, self).__init__()

        self.hidden_size = 120
        self.num_layers = 1 
        self.representation_size = self.hidden_size
        self.output_size = output_size

        # Layers
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True)  # EX4
        num_embeddings, emb_dim = embeddings.shape
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        if concat==False :
            self.linear = nn.Linear(self.representation_size, self.output_size)
        else :
            self.linear = nn.Linear(3*self.representation_size, self.output_size)

    def forward(self, x, lengths) :
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x) # 16, 40, 50

        # Helps the lstm ignore the padded zeros
        # len=4, X[0]:torch.Size([358, 50]), <class 'torch.nn.utils.rnn.PackedSequence'>
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X) #ht lstm size: 4 torch.Size([358, 120]) <class 'torch.nn.utils.rnn.PackedSequence'>

        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True) #16, 33, 120 where 33 could be : 1-40
        # Sentence representation as the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_size).float() 
        for i in range(lengths.shape[ 0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]
        
        if self.concat==True : 
            # mean of ht(for every word)
            representations1 = torch.sum(ht, dim=1)
            for i in range(lengths.shape[0]) :
                representations1[i] = representations1[i] / lengths[i]
            # max of ht in dim 1 (for every word)
            representations2,_ = torch.max(ht, dim=1)
            representations = torch.cat((representations,representations1, representations2), dim=1)  

        


        logits = self.linear(representations) #16, 120
        return logits

class Attention_DNN(nn.Module) :
    def __init__(self, output_size, embeddings) -> None:
        super(Attention_DNN, self).__init__()

        self.output_size = output_size

        # Layers
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True)  # EX4
        num_embeddings, emb_dim = embeddings.shape   
        self.representation_dim = emb_dim   

        # attention 
        self.attention = SelfAttention(emb_dim, batch_first=True, non_linearity='tanh' )

        self.linear = nn.Linear(self.representation_dim, output_size)

    def forward(self, x, lengths) :
        
        embeddings = self.embeddings(x)
        out, scores = self.attention(embeddings, lengths)
        out = self.linear(out)

        return out 


class LSTM_Attention(nn.Module) :
    def __init__(self, output_size, embeddings, concat=False) -> None:
        super(LSTM_Attention, self).__init__()

        self.hidden_size = 120
        self.num_layers = 1 
        self.representation_size = self.hidden_size
        self.output_size = output_size
        self.concat = concat

        # Layers
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True)  # EX4
        num_embeddings, emb_dim = embeddings.shape
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        if concat==False :
            # attention layer
            self.attention = SelfAttention(self.representation_size, batch_first=True, non_linearity='tanh' )
            self.linear = nn.Linear(self.representation_size, self.output_size)
        else :
            # attention layer
            self.attention = SelfAttention(3*self.representation_size, batch_first=True, non_linearity='tanh' )
            self.linear = nn.Linear(3*self.representation_size, self.output_size)

    def forward(self, x, lengths) :
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x) # batch X sequence X emb_dim
        # Helps the lstm ignore the padded zeros
        # len=4, X[0]:torch.Size([358, 50]), <class 'torch.nn.utils.rnn.PackedSequence'>
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X) #ht lstm size: 4 torch.Size([358, 120]) <class 'torch.nn.utils.rnn.PackedSequence'>

        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True) #16, 33, 120 where 33 could be : 1-40
        representations = ht
        if self.concat==True : 
            # mean of ht(for every word)
            representations1 = torch.sum(ht, dim=1)
            for i in range(lengths.shape[0]) :
                representations1[i] = representations1[i] / lengths[i]
            # max of ht in dim 1 (for every word)
            representations2,_ = torch.max(ht, dim=1)
            representations = torch.cat((representations,representations1, representations2), dim=1)  

        weighted_representations, scores = self.attention(representations, lengths)
        logits = self.linear(weighted_representations) #16, 120
        return logits


class SelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = nn.parameter.Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = torch.autograd.Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        print('scores : ', scores.size(),scores.unsqueeze(-1).expand_as(inputs).size() )
        print(scores.unsqueeze(-1).expand_as(inputs))

        # Element wize mult. Attention score is multiplied with each feature 
        # batch X words X emb_dim
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        print(weighted.size())
        # sum the hidden states/words and get batch X emb_dim
        representations = weighted.sum(1).squeeze()
        print(weighted.sum(1).size())
        print(representations.size())

        return representations, scores