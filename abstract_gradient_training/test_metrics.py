"""Bounded test metrics."""

import torch
import torch.nn.functional as F

from abstract_gradient_training import bounded_losses
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import BoundedModel


@torch.no_grad()
def test_mse(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case MSE
    on a batch of the test set using interval arithmetic.

    Args:
        bounded_model (BoundedModel): Bounded version of a pytorch model.
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        targets (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # validate inputs
    if epsilon < 0:
        raise ValueError("Feature poisoning parameter must be non-negative.")
    batch = batch.to(bounded_model.device).type(bounded_model.dtype)
    targets = targets.squeeze() if targets.dim() > 1 else targets
    targets = targets.to(bounded_model.device)
    loss = bounded_losses.BoundedMSELoss(reduction="mean")

    # perform forward passes through the model and loss
    logit_n = bounded_model.forward(batch)
    logit_l, logit_u = bounded_model.bound_forward(batch - epsilon, batch + epsilon)
    mse_n = loss.forward(logit_n, targets)
    mse_l, mse_u = loss.bound_forward(logit_l, logit_u, targets)
    interval_arithmetic.validate_interval(mse_l, mse_u, mse_n, msg="final mse bounds")
    return mse_u.detach().cpu().item(), mse_n.detach().cpu().item(), mse_l.detach().cpu().item()


@torch.no_grad()
def test_accuracy(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case prediction accuracy
    on a batch of the test set using interval arithmetic.

    Args:
        bounded_model (BoundedModel): Bounded version of a pytorch model.
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # validate inputs
    if epsilon < 0:
        raise ValueError("Feature poisoning parameter must be non-negative.")
    batch = batch.to(bounded_model.device).type(bounded_model.dtype)
    labels = labels.to(bounded_model.device).type(torch.int64)
    labels = labels.squeeze() if labels.dim() > 1 else labels
    loss = bounded_losses.BoundedAccuracy(reduction="mean")

    # perform forward passes through the model and loss
    logit_n = bounded_model.forward(batch)
    logit_l, logit_u = bounded_model.bound_forward(batch - epsilon, batch + epsilon)
    acc_n = loss.forward(logit_n, labels)
    acc_l, acc_u = loss.bound_forward(logit_l, logit_u, labels)
    interval_arithmetic.validate_interval(acc_l, acc_u, acc_n, msg="final accuracy bounds")
    return acc_l.detach().cpu().item(), acc_n.detach().cpu().item(), acc_u.detach().cpu().item()


def test_cross_entropy(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case cross entropy loss
    on a batch of the test set using interval arithmetic.

    Args:
        bounded_model (BoundedModel): Bounded version of a pytorch model.
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # validate inputs
    if epsilon < 0:
        raise ValueError("Feature poisoning parameter must be non-negative.")
    batch = batch.to(bounded_model.device).type(bounded_model.dtype)
    labels = labels.to(bounded_model.device).type(torch.int64)
    labels = labels.squeeze() if labels.dim() > 1 else labels

    # nominal, lower and upper bounds for the forward pass
    logit_n = bounded_model.forward(batch)
    logit_l, logit_u = bounded_model.bound_forward(batch - epsilon, batch + epsilon)
    logit_l, logit_n, logit_u = logit_l.squeeze(), logit_n.squeeze(), logit_u.squeeze()

    # compute bounds on the cross entropy loss given the best and worst case logits
    if logit_l.dim() == 1:  # binary classification
        loss = bounded_losses.BoundedBCEWithLogitsLoss()
        ce_n = loss.forward(logit_n, labels)
        ce_l, ce_u = loss.bound_forward(logit_l, logit_u, labels)
    else:  # multi-class classification
        loss = bounded_losses.BoundedCrossEntropyLoss()
        ce_n = loss.forward(logit_n, labels)
        ce_l, ce_u = loss.bound_forward(logit_l, logit_u, labels)

    interval_arithmetic.validate_interval(ce_l, ce_u, ce_n, msg="final ce loss bounds")
    return ce_u.detach().cpu().item(), ce_n.detach().cpu().item(), ce_l.detach().cpu().item()


def certified_predictions(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    epsilon: float = 0.0,
    return_proportion=True,
) -> torch.Tensor:
    """
    Given bounds on the parameters of a neural network, check whether each input in the batch has a constant prediction
    across all parameters in the bounds. If epsilon > 0, the proportion of certified points is certified for the given
    feature poisoning parameter.

    Args:
        bounded_model (BoundedModel): Bounded version of a pytorch model.
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        epsilon (float, optional): Feature poisoning parameter.
        return_proportion (bool, optional): Flag to indicate if the proportion of certified points should be returned or
            the boolean tensor indicating whether each test point is certified.

    Returns:
        torch.Tensor: boolean tensor indicating whether each test point is certified or the proportion of certified
            points.
    """
    # validate input
    if epsilon < 0:
        raise ValueError("Feature poisoning parameter must be non-negative.")
    batch = batch.to(bounded_model.device).type(bounded_model.dtype)

    # get logit bounds
    logit_l, logit_u = bounded_model.bound_forward(batch - epsilon, batch + epsilon)
    logit_l = logit_l.squeeze() if logit_l.dim() > 1 else logit_l
    logit_u = logit_u.squeeze() if logit_u.dim() > 1 else logit_u

    if logit_l.dim() == 1:  # binary classification
        preds_l, preds_u = logit_l > 0, logit_u > 0
        certified_preds = preds_l == preds_u
        return certified_preds.float().mean() if return_proportion else certified_preds

    assert logit_l.dim() == 2, f"Expected shape of (batchsize, num_classes), got {logit_l.shape}"

    # multi-class classification
    outputs_l, outputs_u = interval_arithmetic.propagate_softmax(logit_l, logit_u)

    # find the class with the highest output lower bound
    target_idx = outputs_l.argmax(dim=1)
    target_idx_lb = outputs_l.max(dim=1).values

    # if the highest lower bound is greater than the upper bounds of all the other classes, then the prediction
    # is certified.
    # for the remaining indices, find the class with the highest upper bound
    outputs_u[torch.arange(outputs_u.size(0)), target_idx] = -float("inf")
    other_idx_ub = outputs_u.max(dim=1).values
    certified_preds = target_idx_lb > other_idx_ub

    return certified_preds.float().mean() if return_proportion else certified_preds
