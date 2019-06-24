import torch
import pickle
import numpy as np


class MFCCDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = pickle.load(open(path, 'rb'))
        self.x = torch.from_numpy(data['x']).type('torch.FloatTensor')
        self.y = torch.from_numpy(data['y'])
        assert len(self.x) == len(self.y)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
