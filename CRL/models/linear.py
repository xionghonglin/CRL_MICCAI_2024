from torch import dropout
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_dim=0, out_dim=0, hidden_list = []):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.layers(x)
