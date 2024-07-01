"""Example fully connected network architecture."""

import torch


class FullyConnected(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay):
        layers = [torch.nn.Linear(in_dim, hidden_dim)]
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)
