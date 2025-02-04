"""
Datasets for the privacy paper experiments.
"""

import torch
import torchvision
import torch.utils.data
import sklearn.model_selection
import sklearn.datasets
import numpy as np

from medmnist import OCTMNIST  # python -m pip install git+https://github.com/MedMNIST/MedMNIST.git
import script_utils


def get_blobs(cluster_pos: float, cluster_std: float, batchsize: int, seed: int):
    """Dataloaders for the blobs dataset. Adds polynomial features to the dataset."""
    # get blobs dataset
    stds = [cluster_std, cluster_std]
    centers = np.array([[cluster_pos, cluster_pos], [-cluster_pos, -cluster_pos]])
    x, y = sklearn.datasets.make_blobs(  # type: ignore
        centers=centers, cluster_std=stds, random_state=seed, n_samples=int(batchsize * 1.25), n_features=2
    )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=seed)
    # add quadratic and cubic features
    x_train = np.hstack((x_train, x_train**2, x_train**3))
    x_test = np.hstack((x_test, x_test**2, x_test**3))
    # convert to pytorch tensors
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    return train_dataset, test_dataset


def get_octmnist(exclude_classes=None, balanced=False, encode: torch.nn.Sequential | None = None):
    """
    Get OCT MedMNIST dataset as a binary classification problem of class 3 (normal) vs classes 0, 1, 2.
    """

    # get the datasets
    train_dataset = OCTMNIST(split="train", transform=torchvision.transforms.ToTensor())
    test_dataset = OCTMNIST(split="test", transform=torchvision.transforms.ToTensor())
    train_imgs, train_labels = train_dataset.imgs, train_dataset.labels
    test_imgs, test_labels = test_dataset.imgs, test_dataset.labels

    # filter out excluded classes
    if exclude_classes is not None:
        for e in exclude_classes:
            train_imgs = train_imgs[(train_labels != e).squeeze()]
            train_labels = train_labels[(train_labels != e).squeeze()]
            test_imgs = test_imgs[(test_labels != e).squeeze()]
            test_labels = test_labels[(test_labels != e).squeeze()]

    # convert to a binary classification problem
    train_labels = train_labels != 3  # i.e. 0 = normal, 1 = abnormal
    test_labels = test_labels != 3

    # apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1) / 255
    test_imgs = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1) / 255
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    # balance the training dataset such that the number of samples in each class is equal
    if balanced:
        n_ones = int(train_labels.sum().item())
        n_zeros = len(train_labels) - n_ones
        n_samples = min(n_ones, n_zeros)
        # find the indices of the ones, and then randomly sample n_samples from them
        idx_ones = torch.where(train_labels == 1)[0]
        ones_selection = torch.randperm(n_ones)
        idx_ones = idx_ones[ones_selection][:n_samples]
        # find the indices of the zeros, and then randomly sample n_samples from them
        idx_zeros = torch.where(train_labels == 0)[0]
        zeros_selection = torch.randperm(n_zeros)
        idx_zeros = idx_zeros[zeros_selection][:n_samples]
        idx = torch.cat([idx_ones, idx_zeros])
        train_imgs, train_labels = train_imgs[idx], train_labels[idx]

    if encode:
        with torch.no_grad():
            train_imgs, test_imgs = train_imgs.to(encode[0].weight.device), test_imgs.to(encode[0].weight.device)
            train_imgs = encode(train_imgs)
            test_imgs = encode(test_imgs)

    # form dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_imgs, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_imgs, test_labels)
    return train_dataset, test_dataset


def get_imdb():
    """Dataloaders for the imdb dataset."""
    _, _, data_dir, _ = script_utils.make_dirs()
    X = torch.load(f"{data_dir}/GPT2_IMDB_X_test", weights_only=True)
    y = torch.load(f"{data_dir}/GPT2_IMDB_Y_test", weights_only=True)
    X_train = torch.load(f"{data_dir}/GPT2_IMDB_X_train", weights_only=True)[10000 - 1 :]
    y_train = torch.load(f"{data_dir}/GPT2_IMDB_Y_train", weights_only=True)[10000 - 1 :]
    X = torch.concatenate((X, X_train), axis=0)
    y = torch.concatenate((y, y_train), axis=0)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    return train_dataset, test_dataset
