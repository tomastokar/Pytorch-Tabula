import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class TabulaData(Dataset):
    def __init__(self, data, cat_cols, num_cols, target_col):
        self.n = data.shape[0]
        
        self.cat_features = (
            data
            .loc[:, cat_cols]
            .values
            .astype(np.int64)
        )

        self.num_features = (
            data
            .loc[:, num_cols]
            .values
            .astype(np.float32)
        )

        self.targets = (
            data
            .loc[:, target_col]
            .values
            .astype(np.float32)
        )
    
        self.cat_features = torch.tensor(self.cat_features)
        self.num_features = torch.tensor(self.num_features)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx], self.targets[idx]



class Tabula(nn.Module):
    def __init__(self, cat_nums, emb_dims, no_numeric, linear_sizes, linear_drops = None):
        super().__init__()
        # Assert dropouts
        if linear_drops is not None:
            assert len(linear_sizes) == len(linear_drops)
            dropout = True
            
        # Embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(n, dim)for n, dim in zip(cat_nums, emb_dims)])

        # Linear layers
        layers = []
        no_inputs = sum(emb_dims) + no_numeric
        for i, dim in enumerate(linear_sizes):
            layers.append(nn.Linear(no_inputs, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(no_inputs))
            no_inputs = dim
            # Add dropout
            if dropout:
                layers.append(nn.Dropout(linear_drops[i]))            

        # Finishing layer
        layers.append(nn.Linear(no_inputs, 1))

        # Add to object as a sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, cat_data, num_data):    
        x = [emb(cat_data[:, i]) for i,emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = torch.cat([x, num_data], 1)
        x = self.layers(x)

        return x
