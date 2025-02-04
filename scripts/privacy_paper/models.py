"""Models for the privacy paper experiments."""

import os
from typing import overload, Literal

import torch
import torch.utils.data
import tqdm

import abstract_gradient_training as agt

import script_utils

import datasets


def fully_connected(
    seed: int,
    width: int = 128,
    depth: int = 1,
    in_dim: int = 6,
    out_dim: int = 2,
    dropout: float = 0.0,
) -> torch.nn.Sequential:
    """Initialise the neural network model."""
    torch.manual_seed(seed)

    if depth == 0:
        model = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim))
        if dropout > 0:
            model.append(torch.nn.Dropout(dropout))
        return model

    layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, width)]
    layers.append(torch.nn.ReLU())
    for _ in range(depth - 1):
        layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(width, out_dim))
    model = torch.nn.Sequential(*layers)
    return model


def get_octmnist_pretrained(
    seed: int,
    device: str = "cuda:1",
    pretrain_batchsize: int = 100,
    pretrain_n_epochs: int = 20,
    pretrain_learning_rate: float = 0.001,
) -> torch.nn.Sequential:
    """Get the pretrained OCT MedMNIST model. If it doesn't exist, pretrain one."""
    model_dir = script_utils.make_dirs()[1]
    model_path = f"{model_dir}/medmnist_{seed}_{pretrain_batchsize}_{pretrain_learning_rate}_{pretrain_n_epochs}.ckpt"
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 4, 2, 0),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 4, 1, 0),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(3200, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
    )
    device = torch.device(device)  # type: ignore
    # check if a pre-trained model exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(device)
        return model

    # else train the model
    dataset_pretrain, _ = datasets.get_octmnist(exclude_classes=[2], balanced=True)
    dl_pretrain = torch.utils.data.DataLoader(dataset_pretrain, batch_size=pretrain_batchsize, shuffle=True)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_learning_rate)  # type: ignore
    for _ in (progress_bar := tqdm.trange(pretrain_n_epochs, desc="Epoch")):
        for i, (x, u) in enumerate(dl_pretrain):
            # Forward pass
            u, x = u.to(device), x.to(device)
            output = model(x)
            loss = criterion(output.squeeze().float(), u.squeeze().float())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                progress_bar.set_postfix(loss=loss.item())
    # save the model
    with open(model_path, "wb") as file:
        torch.save(model.state_dict(), file)

    return model
