"""Dataloaders for the 'UCI' datasets."""

import torch
from torch.utils.data import TensorDataset, DataLoader
import uci_datasets  # python -m pip install git+https://github.com/treforevans/uci_datasets.git


def get_dataloaders(train_batchsize, test_batchsize=500, dataset="slice", n_batches=None):
    """
    Get dataloaders for the uci datasets.
    """
    # Check if the dataset name is valid
    if dataset not in uci_datasets.all_datasets:
        raise ValueError(f"Dataset {dataset} not found in UCI datasets.")

    # Get the dataset
    data = uci_datasets.Dataset(dataset)
    x_train, y_train, x_test, y_test = data.get_split(split=0)

    # Convert to float32 and add an extra dimension
    x_train, y_train = x_train.astype("float32")[:, :, None], y_train.astype("float32")[:, :, None]
    x_test, y_test = x_test.astype("float32")[:, :, None], y_test.astype("float32")[:, :, None]

    # Normalise x_train, x_test, y_train, y_test
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
    x_test = (x_test - x_test.min(axis=0)) / (x_test.max(axis=0) - x_test.min(axis=0))
    y_train = (y_train - y_train.min(axis=0)) / (y_train.max(axis=0) - y_train.min(axis=0))
    y_test = (y_test - y_test.min(axis=0)) / (y_test.max(axis=0) - y_test.min(axis=0))

    # Shuffle the data
    idx = (
        torch.randperm(x_train.shape[0])
        if n_batches is None
        else torch.randperm(x_train.shape[0])[: n_batches * train_batchsize]
    )
    x_train = x_train[idx]
    y_train = y_train[idx]
    idx = (
        torch.randperm(x_test.shape[0])
        if n_batches is None
        else torch.randperm(x_test.shape[0])[: n_batches * train_batchsize]
    )
    x_test = x_test[idx]
    y_test = y_test[idx]

    # Convert to tensors and set datatypes
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Form datasets and dataloaders
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=train_batchsize, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batchsize, shuffle=False)

    return train_loader, test_loader
