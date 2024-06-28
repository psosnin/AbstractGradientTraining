"""Test metrics and bounding functions."""

from typing import Callable, Optional
import torch
import torch.nn.functional as F

from abstract_gradient_training import nominal_pass
from abstract_gradient_training.bounds import interval_bound_propagation as ibp


def test_mse(
    param_n: list[torch.Tensor],
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    dl_test: torch.utils.data.DataLoader,
    model: Optional[torch.nn.Sequential] = None,
    transform: Optional[Callable] = None,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case MSE
    on a batch of the test set using interval arithmetic.

    Args:
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        dl_test (torch.utils.data.DataLoader): Test DataLoader
        model (torch.nn.Sequential, optional): Model to transform the input data through. Defaults to None.
        transform (Callable, optional): Function that transforms and bounds the input data for any fixed layers of
                                        the provided model.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # get the test batch and send it to the correct device
    device = param_n[-1].device
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch, targets = next(iter(dl_test))
    batch, targets = batch.to(device).type(param_n[-1].dtype), targets.squeeze().to(device)
    assert targets.dim() == 1, "Targets must be of shape (batchsize, )"

    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform is not None:
        batch_n = transform(batch, model, 0)[0]
        batch_l, batch_u = transform(batch, model, epsilon)
    else:
        batch_n = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch_n - epsilon, batch_n + epsilon
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l, logit_n, logit_u = logit_l.squeeze(), logit_n.squeeze(), logit_u.squeeze()
    targets = targets.reshape(logit_l.shape)
    # calculate best and worst case differences
    diffs_l = logit_l - targets
    diffs_u = logit_u - targets
    best_case = torch.mean(
        torch.zeros_like(diffs_l)  # if the differences span zero, then the best MSE is zero
        + (diffs_l**2) * (diffs_l > 0)  # if the differences are positive, best case MSE is lb
        + (diffs_u**2) * (diffs_u < 0)  # if the differences are negative, best case MSE is ub
    )
    worst_case = torch.mean(torch.maximum(diffs_l**2, diffs_u**2))
    nominal_case = torch.mean((logit_n - targets) ** 2)
    return worst_case.detach().cpu().item(), nominal_case.detach().cpu().item(), best_case.detach().cpu().item()


def test_accuracy(
    param_n: list[torch.Tensor],
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    dl_test: torch.utils.data.DataLoader,
    model: Optional[torch.nn.Sequential] = None,
    transform: Optional[Callable] = None,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case prediction accuracy
    on a batch of the test set using interval arithmetic.

    Args:
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        dl_test (torch.utils.data.DataLoader): Test DataLoader
        model (torch.nn.Sequential, optional): Model to transform the input data through. Defaults to None.
        transform (Callable, optional): Function that transforms and bounds the input data for any fixed layers of
                                        the provided model.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # get the test batch and send it to the correct device
    device = param_n[-1].device
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch, labels = next(iter(dl_test))
    batch, labels = batch.to(device).type(param_n[-1].dtype), labels.squeeze().to(device)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform:
        batch_n = transform(batch, model, 0)[0]
        batch_l, batch_u = transform(batch, model, epsilon)
    else:
        batch_n = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch_n - epsilon, batch_n + epsilon
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l, logit_n, logit_u = logit_l.squeeze(), logit_n.squeeze(), logit_u.squeeze()
    if logit_l.dim() == 1:  # binary classification
        worst_case = (1 - labels) * logit_u + labels * logit_l
        best_case = labels * logit_u + (1 - labels) * logit_l
        y_n = logit_n > 0
        y_worst = worst_case > 0
        y_best = best_case > 0
    else:  # multi-class classification
        # calculate best and worst case output from the network given the parameter bounds
        v1 = F.one_hot(labels, num_classes=logit_l.size(1))
        v2 = 1 - v1
        worst_case = v2 * logit_u + v1 * logit_l
        best_case = v1 * logit_u + v2 * logit_l
        # calculate post-softmax output
        y_n = torch.nn.Softmax(dim=1)(logit_n).argmax(dim=1)
        y_worst = torch.nn.Softmax(dim=1)(worst_case).argmax(dim=1)
        y_best = torch.nn.Softmax(dim=1)(best_case).argmax(dim=1)
    assert y_n.shape == labels.shape
    accuracy = (y_n == labels).float().mean().item()
    max_accuracy = (y_best == labels).float().mean().item()
    min_accuracy = (y_worst == labels).float().mean().item()
    return min_accuracy, accuracy, max_accuracy


def test_cross_entropy(
    param_n: list[torch.Tensor],
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    dl_test: torch.utils.data.DataLoader,
    model: Optional[torch.nn.Sequential] = None,
    transform: Optional[Callable] = None,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case cross entropy loss
    on a batch of the test set using interval arithmetic.

    Args:
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        dl_test (torch.utils.data.DataLoader): Test DataLoader
        model (torch.nn.Sequential, optional): Model to transform the input data through. Defaults to None.
        transform (Callable, optional): Function that transforms and bounds the input data for any fixed layers of
                                        the provided model.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # get the test batch and send it to the correct device
    device = param_n[-1].device
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch, labels = next(iter(dl_test))
    batch, labels = batch.to(device).type(param_n[-1].dtype), labels.squeeze().to(device)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"

    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform:
        batch_n = transform(batch, model, 0)[0]
        batch_l, batch_u = transform(batch, model, epsilon)
    else:
        batch_n = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch_n - epsilon, batch_n + epsilon
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l, logit_n, logit_u = logit_l.squeeze(), logit_n.squeeze(), logit_u.squeeze()

    if logit_l.dim() == 1:  # binary classification
        worst_case_logits = (1 - labels) * logit_u + labels * logit_l
        best_case_logits = labels * logit_u + (1 - labels) * logit_l

        nominal_loss = F.binary_cross_entropy_with_logits(logit_n, labels.float())
        best_case_loss = F.binary_cross_entropy_with_logits(best_case_logits, labels.float())
        worst_case_loss = F.binary_cross_entropy_with_logits(worst_case_logits, labels.float())

    else:  # multi-class classification
        # calculate best and worst case output from the network given the parameter bounds
        v1 = F.one_hot(labels, num_classes=logit_l.size(1))
        v2 = 1 - v1
        worst_case_logits = v2 * logit_u + v1 * logit_l
        best_case_logits = v1 * logit_u + v2 * logit_l

        nominal_loss = F.cross_entropy(logit_n, labels)
        best_case_loss = F.cross_entropy(best_case_logits, labels)
        worst_case_loss = F.cross_entropy(worst_case_logits, labels)

    assert nominal_loss <= worst_case_loss
    assert nominal_loss >= best_case_loss
    return worst_case_loss.item(), nominal_loss.item(), best_case_loss.item()
