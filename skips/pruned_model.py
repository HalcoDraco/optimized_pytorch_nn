import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.dims = [28 * 28, 128, 32, 32, 32, 10]

        self.linear_layers = nn.ModuleList([
            nn.Linear(self.dims[0], self.dims[1]),
            nn.Linear(self.dims[0], self.dims[2]),
            nn.Linear(self.dims[0], self.dims[3]),
            nn.Linear(self.dims[0], self.dims[4]),
            nn.Linear(self.dims[0], self.dims[5]),

            nn.Linear(self.dims[1], self.dims[2]),
            nn.Linear(self.dims[1], self.dims[3]),
            nn.Linear(self.dims[1], self.dims[4]),
            nn.Linear(self.dims[1], self.dims[5]),

            nn.Linear(self.dims[2], self.dims[3]),
            nn.Linear(self.dims[2], self.dims[4]),
            nn.Linear(self.dims[2], self.dims[5]),

            nn.Linear(self.dims[3], self.dims[4]),
            nn.Linear(self.dims[3], self.dims[5]),

            nn.Linear(self.dims[4], self.dims[5]),
        ])

    def forward(self, x):
        num_layers = len(self.dims)
        outs = [None] * (self.layer_selector(num_layers-2, num_layers-1, num_layers) + 1)
        layer_outs = [None] * num_layers

        layer_outs[0] = self.flatten(x)

        for o in range(1, num_layers):
            for i in range(o):
                outs[self.layer_selector(i, o, num_layers)] = F.relu(self.linear_layers[self.layer_selector(i, o, num_layers)](layer_outs[i]))
            layer_outs[o] = outs[self.layer_selector(0, o, num_layers)]
            for i in range(1, o):
                layer_outs[o] = layer_outs[o] + outs[self.layer_selector(i, o, num_layers)]

        return layer_outs[-1]

    @staticmethod
    def layer_selector(input_layer_idx: int, output_layer_idx: int, num_layers: int) -> int:
        return (num_layers*(num_layers-1) - (num_layers-input_layer_idx)*(num_layers-input_layer_idx-1))//2 + (output_layer_idx - input_layer_idx) - 1

    #TODO: Pruning