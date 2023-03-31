import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, signal_length=128, n_out=64, n_layer=2, dropout=True):
        super(MLPModel, self).__init__()
        relu = nn.ReLU()
        dropout = nn.Dropout(p=0.1)

        layers = []
        layers.append(nn.Linear(signal_length, 2*signal_length))
        
        for l in range(n_layer-1):
            layers.append( nn.Linear(2*signal_length, 2*signal_length) )
            layers.append( relu )
            if dropout:
                layers.append( dropout )
        
        layers.append(nn.Linear(2*signal_length, n_out) )

        self.layers = nn.Sequential(*layers)

            
    def forward(self, inputs):
        x = self.layers(inputs)
        return x


    

class CNN_1d(nn.Module):
    def __init__(self, dropout=0.1, n_out=2):
        super(CNN_1d,self).__init__()

        self.net = nn.Sequential(nn.Conv1d(1, 128,kernel_size=8, stride=1),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=5, stride=1),
                                nn.Conv1d(128, 256,kernel_size=5, stride=1),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=5, stride=1),
                                nn.Conv1d(256, 128,kernel_size=3, stride=1),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(1),
                                nn.Flatten(),
                                nn.Linear(128,n_out))

    def forward(self,x):
        return self.net(x)
