"""Dataloaders for the 'blobs' dataset."""

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloaders(batchsize, cluster_pos, cluster_std, random_state):
    """Dataloaders for the blobs dataset. Adds polynomial features to the dataset."""
    # get blobs dataset
    stds = [cluster_std, cluster_std]
    centers = [[cluster_pos, cluster_pos], [-cluster_pos, -cluster_pos]]
    x, y = sklearn.datasets.make_blobs(
        centers=centers, cluster_std=stds, random_state=random_state, n_samples=batchsize * 2, n_features=2
    )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    # add quadratic and cubic features
    x_train = np.hstack((x_train, x_train**2, x_train**3))
    x_test = np.hstack((x_test, x_test**2, x_test**3))
    # convert to pytorch tensors
    train_dataset = TensorDataset(torch.from_numpy(x_train).double(), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test).double(), torch.from_numpy(y_test))
    # prepare dataloaders
    dl = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=False,
    )
    dl_test = DataLoader(dataset=test_dataset, batch_size=500, shuffle=False)
    return dl, dl_test
