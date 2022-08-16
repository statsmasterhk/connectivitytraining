import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class ConnNet(nn.Module):
    def __init__(self, in_dim = 4, out_dim = 2, dropout_prob = 0.3):
        super(ConnNet,self).__init__()
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_dim)

    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.fc4(x)

        if not self.training:
            x = F.softmax(x, dim=1)

        return x