import torch
import torch.nn.functional as F
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training import bound_utils

"""
Functions to compute the partial derivative of loss function with respect to the logits.
"""


def bound_cross_entropy_derivative(logit_l, logit_u, logit_n, labels, k_label_poison=0, poison_target=-1, **kwargs):
    """
    Compute an interval bound for the partial derivative of the cross-entropy loss with respect to the logits.
    If k_poison > 0, we must also allow the labels to be perturbed (epsilon has no effect for classification).
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
        dL_l = y_l - y_t
        dL_n = y_n - y_t
        dL_u = y_u - y_t
    else:  # allow for any of the labels to be poisoned.
        if poison_target == -1:
            y_t_u = 1.0
        else:
            y_t_u = y_t.clone()
            y_t_u[:, poison_target] = 1.0
        dL_l = y_l - y_t_u
        dL_n = y_n - y_t
        dL_u = y_u - 0.0
    bound_utils.validate_interval(dL_l, dL_n)
    bound_utils.validate_interval(dL_n, dL_u)
    return dL_l, dL_u, dL_n


def bound_bce_derivative(logit_l, logit_u, logit_n, labels, k_label_poison=0, **kwargs):
    """
    Compute an interval bound for the partial derivative of the binary cross-entropy loss with respect to the logits.
    If k_poison > 0, we must also allow the labels to be perturbed (epsilon has no effect for classification).
    """
    assert logit_l.dim() == 3, "Logits must be of shape (batch_size, 1, 1) for binary cross-entropy loss."
    assert logit_l.shape[-1] == 1
    assert logit_l.shape[-2] == 1, "Binary cross-entropy loss must have logit size (batch_size, 1, 1)."
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."
    labels = labels.view(-1, 1, 1).type(logit_l.dtype)
    if k_label_poison == 0:
        dL_l = torch.sigmoid(logit_l) - labels
        dL_n = torch.sigmoid(logit_n) - labels
        dL_u = torch.sigmoid(logit_u) - labels
    else:  # we must allow for any of the labels to be poisoned
        dL_l = torch.sigmoid(logit_l) - 1
        dL_n = torch.sigmoid(logit_n) - labels
        dL_u = torch.sigmoid(logit_u) - 0
    bound_utils.validate_interval(dL_l, dL_n)
    bound_utils.validate_interval(dL_n, dL_u)
    return dL_l, dL_u, dL_n


def bound_max_margin_derivative(logit_l, logit_u, logit_n, labels, k_label_poison=0, **kwargs):
    """
    Compute an interval bound for the partial derivative of the max-margin loss with respect to the logits.
    Follows the convention of torch.nn.MultiMarginLoss with p = 1 and margin = 1.
    If k_poison > 0, we must also allow the labels to be perturbed (epsilon has no effect for classification).
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
    dL_l = (((1 - logit_u[idx, labels][:, None] + logit_l) > 0) / logit_l.shape[-2]).type(logit_l.dtype)
    dL_u = (((1 - logit_l[idx, labels][:, None] + logit_u) > 0) / logit_l.shape[-2]).type(logit_l.dtype)
    dL_n = (((1 - logit_n[idx, labels][:, None] + logit_n) > 0) / logit_l.shape[-2]).type(logit_l.dtype)
    # adjust the "label" positions
    dL_l[idx, labels] = 0.0
    dL_u[idx, labels] = 0.0
    dL_n[idx, labels] = 0.0
    tmp = - dL_l.sum(dim=-2)
    dL_l[idx, labels] = - dL_u.sum(dim=-2)
    dL_u[idx, labels] = tmp  # use tmp variable for - dL_l.sum(dim=-2) since the value changes in the line above
    dL_n[idx, labels] = - dL_n.sum(dim=-2)
    bound_utils.validate_interval(dL_l, dL_n)
    bound_utils.validate_interval(dL_n, dL_u)
    return dL_l, dL_u, dL_n


def bound_hinge_derivative(logit_l, logit_u, logit_n, labels, k_label_poison=0, **kwargs):
    """
    Compute an interval bound for the partial derivative of the hinge loss with respect to the logits.
    Follows the convention of torch.nn.HingeEmbeddingLoss but with 0/1 labels instead of -1/1.
    If k_poison > 0, we must also allow the labels to be perturbed (epsilon has no effect for classification).
    NOTE: There are some issues with training this loss, it is recommended to use binary cross-entropy instead.
    """
    assert logit_l.dim() == 3, "Logits must be of shape (batch_size, 1, 1) for binary loss."
    assert logit_l.shape[-1] == 1
    assert logit_l.shape[-2] == 1, "Hinge loss must have logit size (batch_size, 1, 1)."
    assert k_label_poison == 0, "Poisoning is not supported for hinge loss."
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."

    labels = labels.view(-1, 1, 1).type(logit_l.dtype)

    dL_l = 1.0 * (labels == 1) - 1.0 * (labels == 0) * (logit_l < 1)
    dL_n = 1.0 * (labels == 1) - 1.0 * (labels == 0) * (logit_n < 1)
    dL_u = 1.0 * (labels == 1) - 1.0 * (labels == 0) * (logit_u < 1)
    bound_utils.validate_interval(dL_l, dL_n)
    bound_utils.validate_interval(dL_n, dL_u)
    return dL_l.type(logit_l.dtype), dL_u.type(logit_l.dtype), dL_n.type(logit_l.dtype)


def bound_mse_derivative(logit_l, logit_u, logit_n, target, label_epsilon=0.0, **kwargs):
    """
    Compute an interval bound for the partial derivative of the mean squared error loss with respect to the logits.
    If k_poison > 0, we must also allow the labels to be perturbed by up to epsilon
    """
    assert len(logit_l.shape) == 3  # we expect batched inputs
    assert label_epsilon >= 0, "Target epsilon must be greater than 0"
    target = target.view(-1, 1, 1).type(logit_l.dtype)
    assert kwargs.get("poison_target", -1) == -1, "Specifying a poison target is not supported for this loss."

    # calculate the derivative of the loss
    dL_l = (2 * (logit_l - target - label_epsilon))
    dL_n = (2 * (logit_n - target))
    dL_u = (2 * (logit_u - target + label_epsilon))
    bound_utils.validate_interval(dL_l, dL_n)
    bound_utils.validate_interval(dL_n, dL_u)
    return dL_l, dL_u, dL_n
