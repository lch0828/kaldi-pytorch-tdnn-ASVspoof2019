import torch
import torch.nn as nn
from models import FTDNN # this is from the github


class F_TDNN(nn.Module):
    def __init__(self, feature_dim):
        super(F_TDNN, self).__init__()
        self.ftdnn = FTDNN(feature_dim)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.ftdnn(x)
        x = self.fc(x)
        return torch.sigmoid(x) 

