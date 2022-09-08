from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, file):
        self.data = file

    def __getitem__(self, item):
        pass

    def __add__(self, other):
        pass


class MyModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


file = ''
batch_size = 16
epoch = 1000
dataset = MyDataset(file)
dataload = DataLoader(dataset, batch_size, shuffle=True)
