"""Test metrics and bounding functions."""

from collections.abc import Callable
import torch
import torch.nn.functional as F

from abstract_gradient_training import nominal_pass
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounds import interval_bound_propagation as ibp


def test_mse(
    param_l: list[torch.Tensor],
    param_n: list[torch.Tensor],
    param_u: list[torch.Tensor],
    batch: torch.Tensor,
    targets: torch.Tensor,
    *,
    transform: Callable | None = None,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case MSE
    on a batch of the test set using interval arithmetic.

    Args:
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        targets (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # validate order of input parameters
    for p_l, p_n, p_u in zip(param_l, param_n, param_u):
        interval_arithmetic.validate_interval(p_l, p_u, p_n, msg="Input param bounds invalid.")
    # get the test batch and send it to the correct device
    device = param_n[-1].get_device()
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch, targets = batch.to(device).type(param_n[-1].dtype), targets.squeeze().to(device)
    assert targets.dim() == 1, "Targets must be of shape (batchsize, )"

    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform is not None:
        batch_n = transform(batch, 0)[0]
        batch_l, batch_u = transform(batch, epsilon)
    else:
        batch_n = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch_n - epsilon, batch_n + epsilon
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l = logit_l.squeeze()
    logit_n = logit_n.squeeze()
    logit_u = logit_u.squeeze()
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
    param_l: list[torch.Tensor],
    param_n: list[torch.Tensor],
    param_u: list[torch.Tensor],
    batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    transform: Callable | None = None,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case prediction accuracy
    on a batch of the test set using interval arithmetic.

    Args:
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # validate order of input parameters
    for p_l, p_n, p_u in zip(param_l, param_n, param_u):
        interval_arithmetic.validate_interval(p_l, p_u, p_n, msg="Input param bounds invalid.")
    # get the test batch and send it to the correct device
    device = param_n[-1].get_device()
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch = batch.to(device).type(param_n[-1].dtype)
    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.to(device).type(torch.int64)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform:
        batch_n = transform(batch, 0)[0]
        batch_l, batch_u = transform(batch, epsilon)
    else:
        batch_n = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch_n - epsilon, batch_n + epsilon
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l = logit_l.squeeze()
    logit_n = logit_n.squeeze()
    logit_u = logit_u.squeeze()
    if logit_l.dim() == 1:  # binary classification
        worst_case = (1 - labels) * logit_u + labels * logit_l
        best_case = labels * logit_u + (1 - labels) * logit_l
        y_n = (logit_n > 0).to(torch.float32)
        y_worst = (worst_case > 0).to(torch.float32)
        y_best = (best_case > 0).to(torch.float32)
    else:  # multi-class classification
        assert labels.max() < logit_l.shape[-1], "Labels must be in the range of the output logit dimension."
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
    param_l: list[torch.Tensor],
    param_n: list[torch.Tensor],
    param_u: list[torch.Tensor],
    batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    transform: Callable | None = None,
    epsilon: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given bounds on the parameters of a neural network, calculate the best, worst and nominal case cross entropy loss
    on a batch of the test set using interval arithmetic.

    Args:
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        tuple[float, float, float]: worst case, nominal case and best case loss
    """
    # validate order of input parameters
    for p_l, p_n, p_u in zip(param_l, param_n, param_u):
        interval_arithmetic.validate_interval(p_l, p_u, p_n, msg="Input param bounds invalid.")
    # get the test batch and send it to the correct device
    device = param_n[-1].get_device()
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch = batch.to(device).type(param_n[-1].dtype)
    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.to(device).type(torch.int64)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"

    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform:
        batch_n = transform(batch, 0)[0]
        batch_l, batch_u = transform(batch, epsilon)
    else:
        batch_n = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch_n - epsilon, batch_n + epsilon
    # nominal, lower and upper bounds for the forward pass
    *_, logit_n = nominal_pass.nominal_forward_pass(batch_n, param_n)
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l = logit_l.squeeze()
    logit_n = logit_n.squeeze()
    logit_u = logit_u.squeeze()
    if logit_l.dim() == 1:  # binary classification
        worst_case_logits = (1 - labels) * logit_u + labels * logit_l
        best_case_logits = labels * logit_u + (1 - labels) * logit_l

        nominal_loss = F.binary_cross_entropy_with_logits(logit_n, labels.float())
        best_case_loss = F.binary_cross_entropy_with_logits(best_case_logits, labels.float())
        worst_case_loss = F.binary_cross_entropy_with_logits(worst_case_logits, labels.float())

    else:  # multi-class classification
        assert labels.max() < logit_l.shape[-1], "Labels must be in the range of the output logit dimension."
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


def certified_predictions(
    param_l: list[torch.Tensor],
    param_n: list[torch.Tensor],
    param_u: list[torch.Tensor],
    batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    transform: Callable | None = None,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """
    Given bounds on the parameters of a neural network, check whether each input in the batch has a constant prediction
    across all parameters in the bounds. If epsilon > 0, the proportion of certified points is certified for the given
    feature poisoning parameter.

    Args:
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.
        epsilon (float, optional): Feature poisoning parameter.

    Returns:
        torch.Tensor: boolean tensor indicating whether each test point is certified.
    """
    # validate order of input parameters
    for p_l, p_n, p_u in zip(param_l, param_n, param_u):
        interval_arithmetic.validate_interval(p_l, p_u, p_n, msg="Input param bounds invalid.")
    # get the test batch and send it to the correct device
    device = param_n[-1].get_device()
    device = torch.device(device) if device != -1 else torch.device("cpu")
    batch = batch.to(device).type(param_n[-1].dtype)
    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.to(device).type(torch.int64)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # for finetuning, we may need to transform the input through the earlier layers of the network
    if transform:
        batch_l, batch_u = transform(batch, epsilon)
    else:
        batch = batch.view(batch.size(0), -1, 1)
        batch_l, batch_u = batch - epsilon, batch + epsilon
    # nominal, lower and upper bounds for the forward pass
    (*_, logit_l), (*_, logit_u) = ibp.bound_forward_pass(param_l, param_u, batch_l, batch_u)
    logit_l = logit_l.squeeze()
    logit_u = logit_u.squeeze()
    if logit_l.dim() == 1:  # binary classification
        worst_case_logits = (1 - labels) * logit_u + labels * logit_l
        worst_case_preds = worst_case_logits > 0
        best_case_logits = labels * logit_u + (1 - labels) * logit_l
        best_case_preds = best_case_logits > 0
    else:  # multi-class classification TODO: need to double check this logic
        assert labels.max() < logit_l.shape[-1], "Labels must be in the range of the output logit dimension."
        # still need to check this logic for multi-class classification?
        assert logit_l.size(1) <= 2, "Only supported for binary classification for now."
        # calculate best and worst case output from the network given the parameter bounds
        v1 = F.one_hot(labels, num_classes=logit_l.size(1))
        v2 = 1 - v1
        worst_case_logits = v2 * logit_u + v1 * logit_l
        best_case_logits = v1 * logit_u + v2 * logit_l
        # calculate post-softmax output
        worst_case_preds = torch.nn.Softmax(dim=1)(worst_case_logits).argmax(dim=1)
        best_case_preds = torch.nn.Softmax(dim=1)(best_case_logits).argmax(dim=1)
    return worst_case_preds == best_case_preds


def proportion_certified(
    param_l: list[torch.Tensor],
    param_n: list[torch.Tensor],
    param_u: list[torch.Tensor],
    batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    transform: Callable | None = None,
    epsilon: float = 0.0,
) -> float:
    """
    Given bounds on the parameters of a neural network, calculate the proportion of inputs in the test set that have a
    constant prediction across all parameters in the bounds. If epsilon > 0, the proportion of certified points is
    certified for the given feature poisoning parameter.

    See `certified_predictions` for details of arguments.

    Returns:
        float: percentage of the test batch with certified predictions.
    """

    certified_points = certified_predictions(
        param_l, param_n, param_u, batch, labels, transform=transform, epsilon=epsilon
    )
    return certified_points.float().mean().item()
