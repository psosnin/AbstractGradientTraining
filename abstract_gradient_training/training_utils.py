"""Helper functions for certified training."""

from __future__ import annotations

import logging
import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.configuration import AGTConfig


LOGGER = logging.getLogger(__name__)


def grads_helper(
    batch_l: torch.Tensor,
    batch_u: torch.Tensor,
    labels: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    config: AGTConfig,
    label_poison: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Helper function to calculate bounds on the gradient of the loss function with respect to all parameters given the
    input and parameter bounds.

    Args:
        batch_l (torch.Tensor): [fragsize x input_dim x 1] tensor of inputs to the network.
        batch_u (torch.Tensor): [fragsize x input_dim x 1] tensor of inputs to the network.
        labels (torch.Tensor): [fragsize, ] tensor of labels for the inputs.
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        config (AGTConfig): Configuration object for the abstract gradient training module.
        label_poison (bool, optional): Boolean flag to indicate if the labels are being poisoned. Defaults to False.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: List of lower and upper bounds on the gradients.
    """
    labels = labels.squeeze()
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # get config parameters
    bound_kwargs = config.bound_kwargs
    loss_bound_fn = config.loss_bound_fn
    forward_bound_fn = config.forward_bound_fn
    backward_bound_fn = config.backward_bound_fn
    label_epsilon = config.label_epsilon if label_poison else 0.0
    k_label_poison = config.label_k_poison if label_poison else 0
    poison_target = config.poison_target if label_poison else -1
    # forward pass through the network with bounds
    activations_l, activations_u = forward_bound_fn(param_l, param_u, batch_l, batch_u, **bound_kwargs)
    # calculate the first partial derivative of the loss function
    # (pass logit_u in as a dummy for logit_n and ignore dl_n)
    dl_l, dl_u, _ = loss_bound_fn(
        activations_l[-1],  # logit_l
        activations_u[-1],
        activations_u[-1],  # logit_u
        labels,
        k_label_poison=k_label_poison,
        label_epsilon=label_epsilon,
        poison_target=poison_target,
    )
    # compute backwards pass through the network with bounds
    grad_min, grad_max = backward_bound_fn(dl_l, dl_u, param_l, param_u, activations_l, activations_u, **bound_kwargs)

    return grad_min, grad_max


def break_condition(evaluation: tuple[float, float, float]) -> bool:
    """
    Check whether to terminate the certified training loop based on the bounds on the test metric (MSE or Accuracy).

    Args:
        evaluation: tuple of the (worst case, nominal case, best case) evaluation of the test metric.

    Returns:
        bool: True if the training should stop, False otherwise.
    """
    if evaluation[0] <= 0.03 and evaluation[2] >= 0.97:  # worst case accuracy bounds too loose
        LOGGER.warning("Early stopping due to loose bounds")
        return True
    if evaluation[0] >= 1e2:  # worst case MSE too large
        LOGGER.warning("Early stopping due to loose bounds")
        return True
    return False


def get_progress_message(
    network_eval: tuple[float, float, float], param_l: list[torch.Tensor], param_u: list[torch.Tensor]
) -> str:
    """
    Generate a progress message for the certified training loop.

    Args:
        network_eval (tuple[float, float, float]): (worst case, nominal case, best case) evaluation of the test metric.
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].

    Returns:
        str: Progress message for the certified training loop.
    """
    msg = (
        f"Network eval bounds=({network_eval[0]:<4.2g}, {network_eval[1]:<4.2g}, {network_eval[2]:<4.2g}), "
        f"W0 Bound={(param_l[0] - param_u[0]).norm():.3} "
    )

    return msg


def propagate_clipping(
    x_l: torch.Tensor, x: torch.Tensor, x_u: torch.Tensor, gamma: float, method: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Propagate the input through a clipping operation. This function is used to clip the gradients in the
    DP-SGD algorithm.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor.
        x_u (torch.Tensor): Upper bound of the input tensor.
        gamma (float): Clipping parameter.
        method (str): Clipping method, one of ["clamp", "norm"].

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the lower, nominal and upper bounds of the clipped
            input tensor.
    """
    if method == "clamp":
        x_l = torch.clamp(x_l, -gamma, gamma)
        x = torch.clamp(x, -gamma, gamma)
        x_u = torch.clamp(x_u, -gamma, gamma)
    elif method == "norm":
        interval_arithmetic.validate_interval(x_l, x_u, msg="input")
        # compute interval over the norm of the input interval
        norms = x.flatten(1).norm(2, dim=1)
        norms_l, norms_u = interval_arithmetic.propagate_norm(x_l, x_u, p=2)
        interval_arithmetic.validate_interval(norms_l, norms_u, msg="norm")
        # compute an interval over the clipping factor
        clip_factor = (gamma / (norms + 1e-6)).clamp(max=1.0)
        clip_factor_l = (gamma / (norms_u + 1e-6)).clamp(max=1.0)
        clip_factor_u = (gamma / (norms_l + 1e-6)).clamp(max=1.0)
        interval_arithmetic.validate_interval(clip_factor_l, clip_factor_u, msg="clip factor")
        # compute an interval over the clipped input
        x_l, x_u = interval_arithmetic.propagate_elementwise(
            x_l, x_u, clip_factor_l.view(-1, 1, 1), clip_factor_u.view(-1, 1, 1)
        )
        x = x * clip_factor.view(-1, 1, 1)
        interval_arithmetic.validate_interval(x_l, x_u, msg="clipped input")
    else:
        raise ValueError(f"Clipping method {method} not recognised.")
    return x_l, x, x_u
