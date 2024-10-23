"""Utility functions for tighter privacy analysis using AGT."""

import math
import logging
from typing import Literal
from collections.abc import Callable
import torch
import scipy

from abstract_gradient_training import nominal_pass
from abstract_gradient_training import test_metrics


LOGGER = logging.getLogger(__name__)


def noisy_test_accuracy(
    param_n: list[torch.Tensor],
    batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    transform: Callable | None = None,
    noise_level: float | torch.Tensor = 0.0,
    noise_type: str = "laplace",
) -> float:
    """
    Given the nominal parameters of a neural network, calculate the prediction accuracy on a batch of the test set,
    adding the specified noise to the predictions.
    NOTE: For now, this function only supports binary classification via the noise + threshold dp mechanism. This
          should be extended to support multi-class problems via the noisy-argmax mechanism in the future.

    Args:
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.
        noise_level (float | torch.Tensor, optional): Noise level for privacy-preserving predictions using the laplace
            mechanism. Can either be a float or a torch.Tensor of shape (batchsize, ).
        noise_type (str, optional): Type of noise to add to the predictions, one of ["laplace", "cauchy"].

    Returns:
        float: The noisy accuracy of the model on the test set.
    """
    # get the test batch and send it to the correct device
    device = param_n[-1].get_device()
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch = batch.to(device).type(param_n[-1].dtype)

    # validate the labels
    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.to(device).type(torch.int64)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"

    # validate the noise parameters and set up the distribution
    assert noise_type in ["laplace", "cauchy"], f"Noise type must be one of ['laplace', 'cauchy'], got {noise_type}"
    noise_level += 1e-7  # can't set distributions scale to zero
    noise_level = torch.tensor(noise_level) if isinstance(noise_level, float) else noise_level
    noise_level = noise_level.to(device).type(param_n[-1].dtype)  # type: ignore
    noise_level = noise_level.expand(labels.size())
    if noise_type == "laplace":
        noise_distribution = torch.distributions.Laplace(0, noise_level)
    else:
        noise_distribution = torch.distributions.Cauchy(0, noise_level)

    # for finetuning, we may need to transform the input through the earlier layers of the network
    batch_n = transform(batch, 0)[0] if transform else batch.view(batch.size(0), -1, 1)
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    logit_n = logit_n.squeeze()

    # transform 2-logit models to a single output
    if logit_n.shape[-1] == 2:
        logit_n = logit_n[:, 1] - logit_n[:, 0]
    if logit_n.dim() > 1:
        raise NotImplementedError("Noisy accuracy is not supported for multi-class classification.")

    # apply noise + threshold dp mechanisim
    y_n = (logit_n > 0).to(torch.float32).squeeze()
    noise = noise_distribution.sample().to(y_n.device).squeeze()
    assert noise.shape == y_n.shape
    y_n = (y_n + noise) > 0.5
    accuracy = (y_n == labels).float().mean().item()
    return accuracy


def compute_min_uncertified_k(
    batch: torch.Tensor,
    labels: torch.Tensor,
    param_bounds_dict: dict[int, list[list[torch.Tensor]]],
    *,
    transform: Callable | None = None,
) -> torch.Tensor:
    """
    For each point in the input batch, compute the minimum of the given 'k_private' values such that the point is not
    certified within the parameter bounds. param_bounds_dict is a dictionary of the form

        {k_private: [param_l, param_n, param_u]}

    where param_l, param_n, and param_u are the lower, nominal, and upper bounds of the network parameters of the
    linear layers of the network in the form [W1, b1, ..., Wm, bm]. The function returns a tensor of shape (batchsize, )
    with a lower bound on the smallest k_private value for which each point is not certified.

    Args:
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        param_bounds_dict (dict[int, list[list[torch.Tensor]]]): Dictionary of k: [param_l, param_n, param_u] values
            obtained from AGT for varying values of k_private.
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.

    Returns:
        torch.Tensor: Lower bound on the smallest k_private value for which each point is not certified.
    """
    # get a param_n value from the dict, which we will use for validating each param bound.
    param_n_check = next(iter(param_bounds_dict.values()))[1]
    check_flag = False
    # initiate a store for the max k for which we can certify each point
    k_max = torch.zeros(labels.size(0), device=param_n_check[0].device)

    for k, (param_l, param_n, param_u) in param_bounds_dict.items():
        if not all(torch.allclose(a, b) for a, b in zip(param_n, param_n_check)):
            check_flag = True
        # check which points are certified for the given k_private value
        certified_points = test_metrics.certified_predictions(
            param_l, param_n, param_u, batch, labels, transform=transform
        )
        k_max = torch.maximum(k_max, k * certified_points)  # update the max k for each point

    # warn the user if the nominal parameters don't match for all k_private values
    if check_flag:
        LOGGER.warning("Nominal parameters don't match for all k_private: check that you are seeding AGT correctly")

    # k_max now stores the max value of k for which we can certify each point.
    # Therefore the smallest k for which we cannot certify each point is at least k_max + 1.
    return k_max + 1


def compute_smooth_sensitivity(
    beta: float,
    batch: torch.Tensor,
    labels: torch.Tensor,
    param_bounds_dict: dict[int, list[list[torch.Tensor]]],
    *,
    transform: Callable | None = None,
) -> torch.Tensor:
    """
    Compute the smooth sensitivity of the model predictions at each point in the test batch by bounding the following
    maximization problem:

        S(x) = max_{k_private} LS(x, k_private) exp(-2 * beta * k_private)

    where LS(x, k_private) returns 1 if the point x is not certified for k_private, and 0 otherwise. The function is
    upper bounded by exp(-2 * beta * k) for any k with LS(x, k) = 0.

    Args:
        beta (float): The beta-smooth sensitivity parameter.
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        param_bounds_dict (dict[int, list[list[torch.Tensor]]]): Dictionary of k: [param_l, param_n, param_u] values
            obtained from AGT for varying values of k_private.
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.

    Returns:
        torch.Tensor: A bound on the smooth sensitivity of the model predictions at each point in the test batch.
    """
    k_max = compute_min_uncertified_k(batch, labels, param_bounds_dict, transform=transform)
    smooth_sensitivity = torch.exp(-2 * beta * k_max)
    return smooth_sensitivity


def compute_local_epsilons(
    batch: torch.Tensor,
    labels: torch.Tensor,
    param_bounds_dict: dict[int, list[list[torch.Tensor]]],
    epsilon: float,
    delta: float,
    clamp: bool = True,
    *,
    transform: Callable | None = None,
) -> torch.Tensor:
    """
    Assuming noise calibrated to the global sensitivity Lap(1 / epsilon), compute the tighter local epsilon^S values
    for each point in the test batch using the smooth sensitivity of the model predictions.

    Args:
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        param_bounds_dict (dict[int, list[list[torch.Tensor]]]): Dictionary of k: [param_l, param_n, param_u] values
            obtained from AGT for varying values of k_private.
        epsilon (float): Global privacy loss parameter.
        delta (float): Global privacy failure parameter.
        clamp (bool, optional): Whether to clamp the local epsilon^S values to the global epsilon value.
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.

    Returns:
        torch.Tensor: A tensor of local epsilon^S values for each point in the test batch.
    """
    assert epsilon > 0 and 1 > delta > 0, "Epsilon and delta must be positive."
    k_star = compute_min_uncertified_k(batch, labels, param_bounds_dict, transform=transform).cpu().numpy()
    local_epsilons = math.log(2 / delta) * scipy.special.lambertw(2 * epsilon * k_star / math.log(2 / delta)) / k_star
    # convert back to tensor. note that lambertw returns a complex number, but in our case the imaginary part is zero.
    local_epsilons = torch.tensor(local_epsilons.real, dtype=torch.float32, device=batch.device)
    local_epsilons = local_epsilons.clamp(0, epsilon) if clamp else local_epsilons
    return local_epsilons


def get_calibrated_noise_level(
    batch: torch.Tensor,
    labels: torch.Tensor,
    param_bounds_dict: dict[int, list[list[torch.Tensor]]],
    epsilon: float,
    delta: float = 0.0,
    noise_type: Literal["cauchy", "laplace"] = "cauchy",
    *,
    transform: Callable | None = None,
) -> torch.Tensor:
    """
    Compute the noise level calibrated to the smooth sensitivity bounds of each prediction in the batch. There are two
    possible mechanisms:

        - Adding Lap(2 * S(x) / epsilon) gives epsilon-delta dp.
        - Adding Cauchy(6 * S(x) / epsilon) gives epsilon dp.

    Args:
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        param_bounds_dict (dict[int, list[list[torch.Tensor]]]): Dictionary of k: [param_l, param_n, param_u] values
            obtained from AGT for varying values of k_private.
        epsilon (float): Global privacy loss parameter.
        delta (float): Global privacy failure parameter.
        noise_type (Literal["cauchy", "laplace"]): Which noise mechanism to use.
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.

    Returns:
        torch.Tensor: Noise level calibrated to the smooth sensitivity bounds of each prediction in the batch.
    """
    if noise_type == "laplace":
        assert epsilon > 0 and 1 > delta > 0, "Epsilon must be positive."
        beta = epsilon / (2 * math.log(2 / delta))
        smooth_sens = compute_smooth_sensitivity(beta, batch, labels, param_bounds_dict, transform=transform)
        noise_level = 2 * smooth_sens / epsilon
    elif noise_type == "cauchy":
        assert epsilon > 0, "Epsilon must be positive."
        if delta > 0:
            LOGGER.debug("Ignoring delta > 0 for the Cauchy noise mechanism.")
        beta = epsilon / 6 - 1e-7
        smooth_sens = compute_smooth_sensitivity(beta, batch, labels, param_bounds_dict, transform=transform)
        noise_level = 6 * smooth_sens / epsilon
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noise_level
