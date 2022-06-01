from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
from config import MAX_LENGTH
import torch.tensor as tensor

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        self.data = X
        self.labels = y
        self.word2idx = word2idx

        # EX2
        self.encoded_X=[]
        self.tokenized_X = [word_tokenize(data_point) for data_point in X]
        print("10 first training examples:",self.tokenized_X[0:10])

        # length_tmp=[]
        for review in self.tokenized_X:
            temp=[]
            # length_tmp.append(len(review))
            for i,token in enumerate(review):
                if i > MAX_LENGTH:
                    break
                if token in self.word2idx.keys():
                    temp.append(self.word2idx[token])
                else:
                    temp.append(self.word2idx['<unk>'])
                    
            if len(review)< MAX_LENGTH:
                zeros_pad = MAX_LENGTH - len(review)
                for _ in range(zeros_pad):
                    temp.append(0)

            self.encoded_X.append(temp)

            
            

        # print("Mean of lens:", np.mean(length_tmp))
        # print("Median of lens:", np.median(length_tmp))
        # print("Min-max of lens:", np.min(length_tmp),np.max(length_tmp))
        



    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        example = self.encoded_X[index]
        label = self.labels[index]
        length = len(self.tokenized_X[index])

        return tensor(example), tensor(label), tensor(length)
        

