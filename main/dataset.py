import torch
import os


class MFCCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = torch.load(data_path)
        self.label = torch.load(label_path)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.label)

