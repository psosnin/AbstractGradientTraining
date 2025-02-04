"""
Run model training using AGT on the UCI-houseelectric dataset.
"""

import os
import copy
import itertools

import torch
import torch.utils.data

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training import test_metrics
import uci_datasets  # python -m pip install git+https://github.com/treforevans/uci_datasets.git

import script_utils


# configure training parameters
USE_CACHED = True
SEED = 15
NOMINAL_CONFIG = agt.AGTConfig(
    fragsize=10000,
    learning_rate=0.005,
    lr_decay=0.1,
    lr_min=1e-5,
    n_epochs=1,
    device="cuda:0",
    loss="mse",
    log_level="INFO",
)


def get_dataset(batchsize):
    """Get the uci-houseelectric dataset."""
    data = uci_datasets.Dataset("houseelectric")
    x_train, y_train, x_test, y_test = data.get_split(split=0)

    # Normalise the features and labels
    x_train_mu, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_train_mu) / x_train_std
    x_test = (x_test - x_train_mu) / x_train_std
    y_train_min, y_train_range = y_train.min(axis=0), y_train.max(axis=0) - y_train.min(axis=0)
    y_train = (y_train - y_train_min) / y_train_range
    y_test = (y_test - y_train_min) / y_train_range

    # Form datasets and dataloaders
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
    return train_loader, test_loader


class FullyConnected(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay):
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, hidden_dim)]
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


def get_model(hidden_lay=1, hidden_size=50, seed=SEED):
    """Initialise the neural network model."""
    _, model_dir, _, _ = script_utils.make_dirs()
    model_path = f"{model_dir}/uci_{seed}_{hidden_lay}_{hidden_size}.ckpt"
    torch.manual_seed(seed)
    model = FullyConnected(11, 1, hidden_size, hidden_lay).to(NOMINAL_CONFIG.device)
    # check if a pre-trained model exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=NOMINAL_CONFIG.device))
        return model
    # save the model
    with open(model_path, "wb") as file:
        torch.save(model.state_dict(), file)
    model = model.to(NOMINAL_CONFIG.device)
    return model


def get_training_bounds(config, batchsize, hidden_lay=1, hidden_size=50, seed=SEED):
    """Perform a training run with AGT and save the bounds at each iteration."""
    torch.manual_seed(seed)
    config = copy.deepcopy(config)
    results_dir, _, _, _ = script_utils.make_dirs()
    model_id = f"uci_{seed}_{hidden_lay}_{hidden_size}.ckpt"
    results_path = f"{results_dir}/uci_{seed}_{hidden_lay}_{hidden_size}_training"
    model = get_model(hidden_lay, hidden_size, seed)
    bounded_model = IntervalBoundedModel(model, interval_matmul="exact")
    config.metadata = f"{model_id=}, SEED={seed}, {batchsize=}, {bounded_model=}, {hidden_lay=}, {hidden_size=}"
    fname = f"{results_path}_{config.hash()}"
    if os.path.isfile(fname) and USE_CACHED:  # run exists, so return the previous results
        return torch.load(fname, weights_only=True)

    dl_train, dl_test = get_dataset(batchsize)

    torch.manual_seed(seed)

    results = []
    test_iterator = itertools.cycle(dl_test)

    def log(bm):
        """Log the loss bounds at this iteration."""
        test_point, test_labels = next(test_iterator)
        results.append(test_metrics.test_mse(bm, test_point, test_labels))

    iter_count = 0

    def early_stop_callback(bm):
        nonlocal iter_count
        iter_count += 1
        return iter_count >= 150

    config.early_stopping_callback = early_stop_callback
    config.on_iter_start_callback = log

    agt.poison_certified_training(bounded_model, config, dl_train, dl_test)
    torch.save(results, fname)

    return results
