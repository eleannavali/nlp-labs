import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# callbacks = [
#     EarlyStopping(monitor="accuracy/val", mode="max", patience=50),
#     ModelCheckpoint(monitor="accuracy/val", mode="max", save_last=True)
# ]


class TorchDNN(nn.Module):
    """Create a DNN to extract posteriors that can be used for HMM decoding
    Parameters:
        input_dim (int): Input features dimension
        output_dim (int): Number of classes-phonems
        num_layers (int): Number of hidden layers
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer
        hidden_dim (int): Number of neurons in each hidden layer
        dropout_p (float): Dropout probability for regularization
    """
    def __init__(
        self, input_dim, output_dim, num_layers=4, batch_norm=True, hidden_dim=[256,256,128,64], dropout_p=0.2
    ):
        super(TorchDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.fc_block_1=self.__block__(input_dim,hidden_dim[-1],batch_norm,dropout_p)

        self.fc_block_1=self.__block__(input_dim,hidden_dim[0],batch_norm,dropout_p)
        self.fc_block_2=self.__block__(hidden_dim[0],hidden_dim[1],batch_norm,dropout_p)
        self.fc_block_3=self.__block__(hidden_dim[1],hidden_dim[2],batch_norm,dropout_p)
        self.fc_block_4=self.__block__(hidden_dim[2],hidden_dim[3],batch_norm,dropout_p)
        self.classifier = nn.Sequential(
                nn.Linear(hidden_dim[-1], output_dim),
                nn.Softmax()
            )

    def forward(self, x):
        '''
        Forward-pass
        '''
        out= self.fc_block_1(x)
        out= self.fc_block_2(out)
        out= self.fc_block_3(out)
        out= self.fc_block_4(out)

        return self.classifier(out)

    def __block__(self,in_features,out_features,batch_norm,drop=0.3):
        if batch_norm:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=drop)
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(p=drop)
            )

# if __name__ == "__main__":
#     net=TorchDNN(195,900,4)
#     x=np.random.rand(1,195)
#     X=torch.rand(1,195)
#     print(net)
#     net.eval()
#     c=net(X)
#     print(c)