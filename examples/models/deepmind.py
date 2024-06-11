"""Deepmind small architecture from https://arxiv.org/abs/1810.12715"""

import torch

class DeepMindSmall(torch.nn.Sequential):
    def __init__(self, in_channels, out_dim):
        output = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()
        super().__init__(
            torch.nn.Conv2d(in_channels, 16, 4, 2, 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, 1, 0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_dim),
            output
        )
        