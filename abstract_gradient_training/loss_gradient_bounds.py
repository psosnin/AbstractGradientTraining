"""
Functions to compute bounds on the partial derivative of loss function with respect to the logits of the network
using interval arithmetic.
"""

import torch
import torch.nn.functional as F
from abstract_gradient_training import interval_arithmetic


def bound_cross_entropy_derivative(
    logit_l: torch.Tensor,
    logit_u: torch.Tensor,
    logit_n: torch.Tensor,
    labels: torch.Tensor,
    k_label_poison: int = 0,
    poison_target: int = -1,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound for the partial derivative of the cross-entropy loss with respect to the logits.
    If k_poison > 0, we must also account for label poisoning.

    Args:
        logit_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound on the output logits of the network.
        logit_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound on the output logits of the network.
        logit_n (torch.Tensor): [batchsize x output_dim x 1] Nominal output logits of the network.
        labels (torch.Tensor): [batchsize] Tensor of labels for the batch.
        k_label_poison (int, optional): How many labels to allow to be poisoned. Defaults to 0.
        poison_target (int, optional): Which class index to allow label flipping attacks to flip to. If -1, label
                                       attacks can flip to any class. Defaults to -1.

    Returns:
        dl_l (torch.Tensor): lower bound on the partial derivative of the loss with respect to the logits.
        dl_u (torch.Tensor): upper bound on the partial derivative of the loss with respect to the logits.
        dl_n (torch.Tensor): nominal value of the partial derivative of the loss with respect to the logits.
    """

    assert logit_l.dim() == 3, "Logits must be of shape (batch_size, nclasses, 1) for binary cross-entropy loss."
    assert logit_l.shape[-1] == 1
    assert logit_l.shape[-2] > 1, "Cross-entropy loss does not support binary classification, use BCE instead."
    labels = labels.flatten(start_dim=0)  # to allow for labels to be of shape (batch_size,) or (batch_size, 1)
    assert labels.dim() == 1, f"Labels must be of shape (batch_size,) not {labels.shape}"
    # calculate post-softmax output with and without bounds
    y_n = F.softmax(logit_n, dim=-2)
    y_l, y_u = interval_arithmetic.propagate_softmax(logit_l, logit_u)
    # calculate one-hot encoding of the labels
    y_t = F.one_hot(labels, num_classes=y_n.shape[-2])[:, :, None].type(y_n.dtype)
    if k_label_poison == 0:
        # first partial derivative of the loss
        dl_l = y_l - y_t
        dl_n = y_n - y_t
        dl_u = y_u - y_t
    else:  # allow for any of the labels to be poisoned.
        if poison_target == -1:
            y_t_u = 1.0
        else:
            y_t_u = y_t.clone()
            y_t_u[:, poison_target] = 1.0
        dl_l = y_l - y_t_u
        dl_n = y_n - y_t
        dl_u = y_u - 0.0
    interval_arithmetic.validate_interval(dl_l, dl_n)
    interval_arithmetic.validate_interval(dl_n, dl_u)
    return dl_l, dl_u, dl_n


def bound_bce_derivative(
    logit_l: torch.Tensor,
    logit_u: torch.Tensor,
    logit_n: torch.Tensor,
    labels: torch.Tensor,
    k_label_poison: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound for the partial derivative of the binary cross-entropy loss with respect to the logits.
    If k_poison > 0, we must also account for label poisoning.

    Args:
        logit_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound on the output logits of the network.
        logit_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound on the output logits of the network.
        logit_n (torch.Tensor): [batchsize x output_dim x 1] Nominal output logits of the network.
        labels (torch.Tensor): [batchsize] Tensor of labels for the batch.
        k_label_poison (int, optional): How many labels to allow to be poisoned. Defaults to 0.

    Returns:
        dl_l (torch.Tensor): lower bound on the partial derivative of the loss with respect to the logits.
        dl_u (torch.Tensor): upper bound on the partial derivative of the loss with respect to the logits.
        dl_n (torch.Tensor): nominal value of the partial derivative of the loss with respect to the logits.
    """

    assert logit_l.dim() == 3, "Logits must be of shape (batch_size, 1, 1) for binary cross-entropy loss."
    assert logit_l.shape[-1] == 1
    assert logit_l.shape[-2] == 1, "Binary cross-entropy loss must have logit size (batch_size, 1, 1)."
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."
    labels = labels.view(-1, 1, 1).type(logit_l.dtype)
    if k_label_poison == 0:
        dl_l = torch.sigmoid(logit_l) - labels
        dl_n = torch.sigmoid(logit_n) - labels
        dl_u = torch.sigmoid(logit_u) - labels
    else:  # we must allow for any of the labels to be poisoned
        dl_l = torch.sigmoid(logit_l) - 1
        dl_n = torch.sigmoid(logit_n) - labels
        dl_u = torch.sigmoid(logit_u) - 0
    interval_arithmetic.validate_interval(dl_l, dl_n)
    interval_arithmetic.validate_interval(dl_n, dl_u)
    return dl_l, dl_u, dl_n


def bound_max_margin_derivative(
    logit_l: torch.Tensor,
    logit_u: torch.Tensor,
    logit_n: torch.Tensor,
    labels: torch.Tensor,
    k_label_poison: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound for the partial derivative of the max-margin loss with respect to the logits.
    Follows the convention of torch.nn.MultiMarginLoss with p = 1 and margin = 1.
    If k_poison > 0, we must also account for label poisoning.

    Args:
        logit_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound on the output logits of the network.
        logit_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound on the output logits of the network.
        logit_n (torch.Tensor): [batchsize x output_dim x 1] Nominal output logits of the network.
        labels (torch.Tensor): [batchsize] Tensor of labels for the batch.
        k_label_poison (int, optional): How many labels to allow to be poisoned. Defaults to 0.

    Returns:
        dl_l (torch.Tensor): lower bound on the partial derivative of the loss with respect to the logits.
        dl_u (torch.Tensor): upper bound on the partial derivative of the loss with respect to the logits.
        dl_n (torch.Tensor): nominal value of the partial derivative of the loss with respect to the logits.
    """
    labels = labels.flatten(start_dim=0)
    assert len(logit_l.shape) == 3, "Logits must be of shape (batch_size, 1, 1) for binary cross-entropy loss."
    assert logit_l.shape[-2] > 1, "Max-margin loss does not support binary classification, use BCE instead."
    assert labels.dim() == 1, "Labels must be of shape (batch_size,)."
    assert k_label_poison == 0, "Poisoning is not supported for max-margin loss."
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."
    # get arange index
    idx = torch.arange(logit_l.shape[0], device=logit_l.device)
    # compute ((1 - x[true] + x[i]) > 0) / n
    dl_l = (((1 - logit_u[idx, labels][:, None] + logit_l) > 0) / logit_l.shape[-2]).type(logit_l.dtype)
    dl_u = (((1 - logit_l[idx, labels][:, None] + logit_u) > 0) / logit_l.shape[-2]).type(logit_l.dtype)
    dl_n = (((1 - logit_n[idx, labels][:, None] + logit_n) > 0) / logit_l.shape[-2]).type(logit_l.dtype)
    # adjust the "label" positions
    dl_l[idx, labels] = 0.0
    dl_u[idx, labels] = 0.0
    dl_n[idx, labels] = 0.0
    tmp = -dl_l.sum(dim=-2)
    dl_l[idx, labels] = -dl_u.sum(dim=-2)
    dl_u[idx, labels] = tmp
    dl_n[idx, labels] = -dl_n.sum(dim=-2)
    interval_arithmetic.validate_interval(dl_l, dl_n)
    interval_arithmetic.validate_interval(dl_n, dl_u)
    return dl_l, dl_u, dl_n


def bound_hinge_derivative(
    logit_l: torch.Tensor,
    logit_u: torch.Tensor,
    logit_n: torch.Tensor,
    labels: torch.Tensor,
    k_label_poison: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound for the partial derivative of the hinge loss with respect to the logits.
    Follows the convention of torch.nn.HingeEmbeddingLoss but with 0/1 labels instead of -1/1.
    NOTE: There are some issues with training this loss, it is recommended to use binary cross-entropy instead.
    If k_poison > 0, we must also account for label poisoning.

    Args:
        logit_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound on the output logits of the network.
        logit_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound on the output logits of the network.
        logit_n (torch.Tensor): [batchsize x output_dim x 1] Nominal output logits of the network.
        labels (torch.Tensor): [batchsize] Tensor of labels for the batch.
        k_label_poison (int, optional): How many labels to allow to be poisoned. Defaults to 0.

    Returns:
        dl_l (torch.Tensor): lower bound on the partial derivative of the loss with respect to the logits.
        dl_u (torch.Tensor): upper bound on the partial derivative of the loss with respect to the logits.
        dl_n (torch.Tensor): nominal value of the partial derivative of the loss with respect to the logits.
    """

    assert logit_l.dim() == 3, "Logits must be of shape (batch_size, 1, 1) for binary loss."
    assert logit_l.shape[-1] == 1
    assert logit_l.shape[-2] == 1, "Hinge loss must have logit size (batch_size, 1, 1)."
    assert k_label_poison == 0, "Poisoning is not supported for hinge loss."
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."

    labels = labels.view(-1, 1, 1).type(logit_l.dtype)

    dl_l = 1.0 * (labels == 1) - 1.0 * (labels == 0) * (logit_l < 1)
    dl_n = 1.0 * (labels == 1) - 1.0 * (labels == 0) * (logit_n < 1)
    dl_u = 1.0 * (labels == 1) - 1.0 * (labels == 0) * (logit_u < 1)
    interval_arithmetic.validate_interval(dl_l, dl_n)
    interval_arithmetic.validate_interval(dl_n, dl_u)
    return dl_l.type(logit_l.dtype), dl_u.type(logit_l.dtype), dl_n.type(logit_l.dtype)


def bound_mse_derivative(
    logit_l: torch.Tensor,
    logit_u: torch.Tensor,
    logit_n: torch.Tensor,
    target: torch.Tensor,
    label_epsilon: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound for the partial derivative of the mean squared error loss with respect to the logits.
    Follows the convention of torch.nn.MultiMarginLoss with p = 1 and margin = 1.
    If label_epsilon > 0, we must also allow the labels to be perturbed by up to epsilon

    Args:
        logit_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound on the output logits of the network.
        logit_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound on the output logits of the network.
        logit_n (torch.Tensor): [batchsize x output_dim x 1] Nominal output logits of the network.
        labels (torch.Tensor): [batchsize] Tensor of labels for the batch.
        label_epsilon (float, optional): Max perturbation to the targets in the l-inf norm. Defaults to 0.

    Returns:
        dl_l (torch.Tensor): lower bound on the partial derivative of the loss with respect to the logits.
        dl_u (torch.Tensor): upper bound on the partial derivative of the loss with respect to the logits.
        dl_n (torch.Tensor): nominal value of the partial derivative of the loss with respect to the logits.
    """

    assert len(logit_l.shape) == 3  # we expect batched inputs
    assert label_epsilon >= 0, "Target epsilon must be greater than 0"
    target = target.view(-1, 1, 1).type(logit_l.dtype)
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."

    # calculate the derivative of the loss
    dl_l = 2 * (logit_l - target - label_epsilon)
    dl_n = 2 * (logit_n - target)
    dl_u = 2 * (logit_u - target + label_epsilon)
    interval_arithmetic.validate_interval(dl_l, dl_n)
    interval_arithmetic.validate_interval(dl_n, dl_u)
    return dl_l, dl_u, dl_n
