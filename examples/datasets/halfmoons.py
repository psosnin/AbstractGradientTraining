"""Dataloaders for the halfmoons dataset."""

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloaders(train_batchsize, test_batchsize=500, random_state=0, noise=0.1, n_samples=3000, sep=0.2):
    """
    Get dataloaders for the halfmoons dataset using sklearn.datasets, and adding quadratic and cubic features.
    """
    # get the dataset
    x, y = sklearn.datasets.make_moons(noise=noise, random_state=random_state, n_samples=n_samples)
    # add separation to the moons
    x[y == 0, 1] += sep
    # split the dataset and add polynomial features
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = np.hstack((x_train, x_train**2, x_train**3))
    x_test = np.hstack((x_test, x_test**2, x_test**3))
    # form dataloaders
    dl_train = DataLoader(
        dataset=TensorDataset(torch.from_numpy(x_train).double(), torch.from_numpy(y_train)),
        batch_size=train_batchsize,
        shuffle=False,
    )
    dl_test = DataLoader(
        dataset=TensorDataset(torch.from_numpy(x_test).double(), torch.from_numpy(y_test)),
        batch_size=test_batchsize,
        shuffle=False,
    )
    return dl_train, dl_test
