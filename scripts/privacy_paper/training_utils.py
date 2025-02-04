"""
Utilities for training models and ensemebles using vanilla pytorch or abstract gradient training
while hashing results from previous runs.
"""

# %%

import copy
import os
import logging
import hashlib
from collections.abc import Callable

import tqdm
import torch
import torch.utils.data

import abstract_gradient_training as agt

import script_utils

LOGGER = logging.getLogger(__name__)
USE_CACHE = True  # global flag to enable/disable caching of results.


class Config(agt.AGTConfig):
    """Extended configuration object for experiments that stores a couple additional fields."""

    batchsize: int
    seed: int


def hash_model(
    model: torch.nn.Sequential | agt.bounded_models.BoundedModel,
) -> str:
    """
    Hashes the model (architecture not weights) to a string.
    """
    return hashlib.md5(str(model).encode()).hexdigest()


def train_with_config(
    model: torch.nn.Sequential,
    config: Config,
    dataset_train: torch.utils.data.Dataset,
    cache_path: str = "",
) -> torch.nn.Sequential:
    """
    Train a model with the given configuration and dataset.
    """
    model = copy.deepcopy(model)
    fname = f"{script_utils.make_dirs()[1]}/{cache_path}"

    if os.path.isfile(fname) and USE_CACHE and cache_path:
        LOGGER.debug("Loading cached model from %s", fname)
        model.load_state_dict(torch.load(fname, weights_only=True))
        return model

    LOGGER.debug("Training model with config %s", config)

    # set up optimizer
    assert config.optimizer in ["SGD", "SGDM"]
    assert config.l1_reg == 0
    momentum = config.optimizer_kwargs.pop("momentum", 0.0)
    dampening = config.optimizer_kwargs.pop("dampening", 0.0)
    nesterov = config.optimizer_kwargs.pop("nesterov", False)
    l2_reg = config.l2_reg
    lr = config.learning_rate
    optimizer = torch.optim.SGD(  # type: ignore
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=l2_reg,
        nesterov=nesterov,
        dampening=dampening,
    )

    # set up loss
    if config.loss == "cross_entropy":  # binary classification with 2 output logits
        criterion = torch.nn.CrossEntropyLoss()
    elif config.loss == "binary_cross_entropy":  # binary classification with 1 output logit
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function {config.loss}")

    # set up dataloader
    n_epochs = config.n_epochs
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batchsize, shuffle=True)
    device = torch.device(config.device)
    model = model.to(device)
    # train the model
    torch.manual_seed(config.seed)
    for epoch in range(n_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            if config.loss == "cross_entropy":
                loss = criterion(y_pred, y)
            else:  # config.loss == "binary_cross_entropy":
                loss = criterion(y_pred.squeeze().float(), y.squeeze().float())
            loss.backward()
            optimizer.step()
        LOGGER.debug("Epoch %d, loss: %f", epoch, loss.item())  # type: ignore

    # save the model
    if cache_path:
        torch.save(model.state_dict(), fname)

    return model


def train_with_config_agt(
    model: torch.nn.Sequential,
    config: Config,
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset | None = None,
    cache_path: str = "",
) -> agt.bounded_models.BoundedModel:
    """
    Train a model with the given configuration and dataset using abstract gradient training.
    """
    model = copy.deepcopy(model)
    fname = f"{script_utils.make_dirs()[1]}/{cache_path}_agt"

    # wrap the model using AGT
    bounded_model = agt.bounded_models.IntervalBoundedModel(model)

    if os.path.isfile(fname) and USE_CACHE and cache_path:
        LOGGER.debug("Loading cached AGT model from %s", fname)
        try:
            bounded_model.load_params(fname)
        except Exception as e:
            print("Failed to load model", e)
            print(fname)
        return bounded_model

    assert not config.k_unlearn and not config.k_poison and not config.label_k_poison
    assert config.noise_multiplier == 0.0
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batchsize, shuffle=True)
    dataloader_test = (
        torch.utils.data.DataLoader(dataset_test, batch_size=10000, shuffle=True) if dataset_test else None
    )
    torch.manual_seed(config.seed)
    agt.privacy_certified_training(bounded_model, config, dataloader, dl_val=dataloader_test)
    if cache_path:
        bounded_model.save_params(fname)
    return bounded_model


def sweep_k_values_with_agt(
    k_values: list[int],
    model: torch.nn.Sequential,
    config: Config,
    dataset_train: torch.utils.data.Dataset,
    cache_path: str = "",
    quiet=False,
    pbar=None,
) -> dict[int, agt.bounded_models.BoundedModel]:
    """
    Train a model using AGT for each k in k_values and return a dictionary of k: trained bounded models.
    """

    test_point, test_label = dataset_train[:100]
    test_point, test_label = test_point.to(config.device), test_label.to(config.device)

    k_values = sorted(k_values)

    bounded_model_dict = {}
    warn_flag = 0
    for k in (pbar2 := tqdm.tqdm(k_values, disable=quiet)):
        if k >= len(dataset_train):  # type: ignore
            if warn_flag == 0:
                LOGGER.debug("Skipping k=%d since it is larger than the dataset size", k)
                warn_flag = 1
            continue

        config.k_private = k
        cache_path_k = f"{cache_path}_{k}" if cache_path else cache_path
        bounded_model_dict[k] = train_with_config_agt(model, config, dataset_train, cache_path=cache_path_k)
        # check the accuracy of the model
        certified = agt.test_metrics.certified_predictions(bounded_model_dict[k], test_point)
        if certified <= 0.02:
            LOGGER.debug("Model with k=%d is not certified, skipping the rest", k)
            break
        pbar2.set_postfix({"certified": float(certified)})
        if pbar:
            pbar.set_postfix({"k": k, "certified": float(certified)})

    return bounded_model_dict


def train_ensemble(
    T: int,
    model_init_fn: Callable[[int], torch.nn.Sequential],
    config: Config,
    dataset_train: torch.utils.data.TensorDataset,
    cache_path: str = "",
) -> list[torch.nn.Sequential]:
    """
    Train an ensemble of models with the given configuration and dataset.
    """

    ensemble = []
    indices = torch.randperm(len(dataset_train), generator=torch.Generator().manual_seed(config.seed))
    for t in tqdm.trange(T):
        model = model_init_fn(config.seed + t)
        cache_path_t = f"{cache_path}_{t}"
        dataset_train_t = torch.utils.data.Subset(dataset_train, indices[t::T])  # type: ignore
        ensemble.append(train_with_config(model, config, dataset_train_t, cache_path_t))
    return ensemble


def train_ensemble_agt(
    T: int,
    k_values: list[int],
    model_init_fn: Callable[[int], torch.nn.Sequential],
    config: Config,
    dataset_train: torch.utils.data.TensorDataset,
    cache_path: str = "",
    quiet=False,
) -> list[dict[int, agt.bounded_models.BoundedModel]]:
    """
    Trains an ensemble of models using AGT. Each mode trained with AGT has bounds for each value of k in k_values.
    """
    ensemble = []
    indices = torch.randperm(len(dataset_train), generator=torch.Generator().manual_seed(config.seed))
    for t in (pbar := tqdm.trange(T, disable=quiet)):
        model = model_init_fn(config.seed + t)
        cache_path_t = f"{cache_path}_{t}_of_{T}"
        dataset_train_t = torch.utils.data.Subset(dataset_train, indices[t::T])  # type: ignore
        ensemble.append(
            sweep_k_values_with_agt(k_values, model, config, dataset_train_t, cache_path_t, quiet=True, pbar=pbar)
        )
    return ensemble


def train_dp_sgd_baseline(
    model: torch.nn.Sequential,
    epsilon: float,
    delta: float,
    config: Config,
    dataset_train: torch.utils.data.Dataset,
    cache_path: str,
):
    """Train a baseline model using dpsgd with the given epsilon privacy budget."""
    import opacus  # lazy import to avoid opacus screwing up logging.

    model = copy.deepcopy(model)
    fname = f"{script_utils.make_dirs()[1]}/{cache_path}_dp_sgd_{epsilon}_{delta}, {config.hash()}"

    if os.path.isfile(fname) and USE_CACHE:  # run exists, so return the previous results
        state_dict, epsilon, delta = torch.load(fname)
        model.load_state_dict(state_dict, strict=False)
        return model, epsilon, delta

    # set up optimizer
    assert config.optimizer in ["SGD", "SGDM"]
    assert config.l1_reg == 0
    momentum = config.optimizer_kwargs.pop("momentum", 0.0)
    dampening = config.optimizer_kwargs.pop("dampening", 0.0)
    nesterov = config.optimizer_kwargs.pop("nesterov", False)
    l2_reg = config.l2_reg
    lr = config.learning_rate
    optimizer = torch.optim.SGD(  # type: ignore
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=l2_reg,
        nesterov=nesterov,
        dampening=dampening,
    )

    # set up loss
    if config.loss == "cross_entropy":  # binary classification with 2 output logits
        criterion = torch.nn.CrossEntropyLoss()
    elif config.loss == "binary_cross_entropy":  # binary classification with 1 output logit
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function {config.loss}")

    # set up dataloader
    n_epochs = config.n_epochs
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batchsize, shuffle=True)
    device = torch.device(config.device)
    model = model.to(device)

    # set up the privacy engine
    privacy_engine = opacus.PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(  # type: ignore
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=config.clip_gamma,
        epochs=n_epochs,
    )

    # train the model
    torch.manual_seed(config.seed)
    for epoch in range(n_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            if config.loss == "cross_entropy":
                loss = criterion(y_pred, y)
            else:  # config.loss == "binary_cross_entropy":
                loss = criterion(y_pred.squeeze().float(), y.squeeze().float())
            loss.backward()
            optimizer.step()
        LOGGER.debug("Epoch %d, loss: %f", epoch, loss.item())  # type: ignore

    epsilon = privacy_engine.accountant.get_epsilon(delta=delta)

    # remove the dp-sgd wrapper
    model = model._module

    # save the model and the privacy parameters
    torch.save((model.state_dict(), epsilon, delta), fname)
    return model, epsilon, delta


# %%
if __name__ == "__main__":

    import datasets
    import models

    config = Config(
        learning_rate=2.0,
        l2_reg=0.02,
        n_epochs=2,
        device="cuda:1",
        loss="cross_entropy",
        lr_decay=0.6,
        lr_min=1e-3,
        clip_gamma=0.08,
        log_level="WARNING",
        seed=0,
        batchsize=2000,
    )

    ks_privacy = list(range(0, 200, 1))
    dataset_train_far, dataset_test_far = datasets.get_blobs(1.25, 0.35, 5000, 0)
    model = models.fully_connected(0)

    cache_name = f"blobs_far_{config.hash()}"

    regular_model = train_with_config(
        model,
        config,
        dataset_train_far,
        cache_name,
    )

    agt_model = train_with_config_agt(
        model,
        config,
        dataset_train_far,
        cache_path=cache_name,
    )

    agt_sweep = sweep_k_values_with_agt(
        ks_privacy,
        model,
        config,
        dataset_train_far,
        cache_name,
    )

    ensemble = train_ensemble(
        5,
        models.fully_connected,
        config,
        dataset_train_far,
        cache_name,
    )

    ensemble_agt = train_ensemble_agt(
        5,
        ks_privacy,
        models.fully_connected,
        config,
        dataset_train_far,
        cache_name,
    )
