import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self,feature_dim, window_size):
        super(TDNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=32, kernel_size=3, stride=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, dilation=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, dilation=4)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256 * (window_size-20), 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # permute to (Batch, input_dim, input_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1) # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x) 

