import torch
import torch.nn as nn
import torch.optim as optim
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
        self, input_dim, output_dim, num_layers=2, batch_norm=True, hidden_dim=[256], dropout_p=0.2
    ):
        super(TorchDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_block=[]
        for layer in range(num_layers):
            if layer ==0:
                self.fc_block.append(self.__block__(input_dim,hidden_dim[layer],batch_norm,dropout_p))
            else:
                self.fc_block.append(self.__block__(hidden_dim[layer-1],hidden_dim[layer],batch_norm,dropout_p))
        self.classifier = nn.Sequential(
                nn.Linear(hidden_dim[-1], output_dim),
                nn.Softmax()
            )

    def forward(self, x):
        '''
        Forward-pass
        '''
        for block in self.fc_block:
            out = block(x)
            x=out
        return self.classifier(x)

    def __block__(in_features,out_features,batch_norm,drop=0.3):
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


    def train (self, X_train, y_train, X_val, y_val,optimizer=optim.Adam(lr=0.001), loss=nn.CrossEntropyLoss(), epochs=10):
        running_loss=0
        last_loss=0
        for epoch in epochs:
            