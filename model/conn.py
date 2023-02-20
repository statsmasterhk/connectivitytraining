"""A Direct copy of class "Conn" from Tracking/Models/Conn.py
"""
from torch import nn
import torch.nn.functional as F

class ConnNet(nn.Module):

    """ A Connectivity Network
    expecting intput:
    batch * [feature dist,dx,dy,frame frame difference divided by 2]
    with shape: (batch, in_dim)

    expecting output:
    batch * [Similarity score, 1-Similarity score]
    with shape: (batch, out_dim)

    
    Attributes:
        dropout_prob (float): Dropout probability
        fc1 (torch.nn.layer): Fully connected layer, input shape: (batch, in_dim), output shape (batch, 128)
        fc2 (torch.nn.layer): Fully connected layer, input shape: (batch, 128), output shape (batch, 128)
        fc3 (torch.nn.layer): Fully connected layer, input shape: (batch, 128), output shape (batch, 128)
        fc4 (torch.nn.layer): Final Fully connected layer, input shape: (batch, 128), output shape (batch, out_dim)
    """
    
    def __init__(self, in_dim = 4, out_dim = 2, dropout_prob = 0.3):
        """Initialise model with parameters (input_dimension,output_dimension,dropout_probability)
        
        Args:
            in_dim (int, optional): input (default = 4)
            out_dim (int, optional): out dimension (default = 2)
            dropout_prob (float, optional): Dropout probability (default = 0.3)
        """
        super(ConnNet,self).__init__()
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_dim)

    
    def forward(self, inputs):
        """ Network forward function
        
        Args:
            inputs (Tensor): shape:(batch, in_dim)
        
        Returns:
            tensor (Tensor): shape:(batch, out_dim)
        """
        x = F.relu(self.fc1(inputs))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.fc4(x)

        if not self.training:
            x = F.softmax(x, dim=1)

        return x