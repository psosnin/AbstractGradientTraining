"""Bounded version of the torch.nn.CrossEntropyLoss loss function."""

import logging
from typing import Literal

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_losses import BoundedLoss

LOGGER = logging.getLogger(__name__)


class BoundedCrossEntropyLoss(BoundedLoss):
    """
    Bounded version of the torch.nn.CrossEntropyLoss() loss function.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        """
        Args:
            reduction (str, optional): Specifies the reduction to apply to the output.
        """
        self.reduction = reduction
        if reduction not in ["sum", "mean", "none"]:
            raise ValueError(f"Reduction {reduction} not recognised.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Nominal forward pass through the loss function.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss values.
        """
        # handle shapes and reshaping
        targets = targets.squeeze() if targets.dim() > 1 else targets
        inputs = inputs.squeeze() if inputs.dim() > 2 else inputs
        # check the reshaping is correct
        assert inputs.dim() == 2, f"Expected shape of (batchsize, num_classes), got {inputs.shape}"
        assert targets.dim() == 1, f"Expected shape of (batchsize,), got {targets.shape}"
        return torch.nn.functional.cross_entropy(inputs, targets, reduction=self.reduction)

    def backward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Return the gradients of the loss wrt inputs.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss gradient values.
        """
        input_shape = inputs.size()  # store the original shape of the input
        targets = targets.squeeze() if targets.dim() > 1 else targets
        inputs = inputs.squeeze() if inputs.dim() > 2 else inputs
        assert inputs.dim() == 2, f"Expected shape of (batchsize, num_classes), got {inputs.shape}"
        assert targets.dim() == 1, f"Expected shape of (batchsize,), got {targets.shape}"

        # calculate the ce gradient
        n_classes = inputs.size(1)
        softmax_outputs = torch.nn.functional.softmax(inputs, dim=1)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=n_classes)
        assert softmax_outputs.shape == one_hot_targets.shape, f"{one_hot_targets.shape} != {softmax_outputs.shape}"
        grad = (softmax_outputs - one_hot_targets).reshape(input_shape)
        if self.reduction == "mean":
            LOGGER.warning("AGT processes grads in fragments, so using 'reduction=mean' is not recommended.")
            LOGGER.warning("Specify 'reduction=none' and normalise manually to ensure correct scaling.")
            return grad * n_classes / grad.nelement()
        return grad

    def bound_forward(
        self,
        inputs_l: torch.Tensor,
        inputs_u: torch.Tensor,
        targets: torch.Tensor,
        *,
        label_k_poison: int = 0,
        label_epsilon: float = 0.0,
        poison_target_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bounded forward pass through the loss function. The forward pass does not support bounding wrt label poisoning.

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Not supported on the CE forward pass.
            label_epsilon (float, optional): Not supported on the CE forward pass.
            poison_target_idx (int, optional): Not supported on the CE forward pass.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the loss values.
        """
        if label_epsilon > 0.0 or label_k_poison != 0 or poison_target_idx >= 0:
            raise NotImplementedError("Label poisoning not implemented for cross-entropy loss forward pass.")
        targets = targets.squeeze() if targets.dim() > 1 else targets
        inputs_l = inputs_l.squeeze() if inputs_l.dim() > 2 else inputs_l
        inputs_u = inputs_u.squeeze() if inputs_u.dim() > 2 else inputs_u
        # check the reshaping is correct
        assert inputs_l.dim() == 2, f"Expected shape of (batchsize, num_classes), got {inputs_l.shape}"
        assert inputs_u.dim() == 2, f"Expected shape of (batchsize, num_classes), got {inputs_u.shape}"
        assert targets.dim() == 1, f"Expected shape of (batchsize,), got {targets.shape}"
        if targets.max() >= inputs_l.size(1):
            raise ValueError("Labels must be in the range of the output logit dimension.")

        # calculate bounds on the loss
        assert label_k_poison == 0
        one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=inputs_l.size(1)).to(inputs_l.device)
        worst_case_logits = (1 - one_hot_labels) * inputs_u + one_hot_labels * inputs_l
        best_case_logits = one_hot_labels * inputs_u + (1 - one_hot_labels) * inputs_l
        loss_l = torch.nn.functional.cross_entropy(best_case_logits, targets, reduction=self.reduction)
        loss_u = torch.nn.functional.cross_entropy(worst_case_logits, targets, reduction=self.reduction)
        interval_arithmetic.validate_interval(loss_l, loss_u, msg="loss bound")
        return loss_l, loss_u

    def bound_backward(
        self,
        inputs_l: torch.Tensor,
        inputs_u: torch.Tensor,
        targets: torch.Tensor,
        *,
        label_k_poison: int = 0,
        label_epsilon: float = 0.0,
        poison_target_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounds on the gradients of the loss function with respect to the inputs to the loss function. If a
        poisoning attack is specified, we also compute bounds wrt the poisoning attack with the following parameters:

            - label_k_poison: Maximum number of data-points with poisoned targets.
            - poison_target_idx: Target class for the poisoning attack. If -1, then the attacker may flip labels to any
                class. Otherwise, the attacker may only flip labels to the class with the specified index.

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Maximum number of data-points with poisoned targets.
            label_epsilon (float, optional): Not applicable for classification.
            poison_target_idx (int, optional): Target class for the poisoning attack.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the gradients of the loss.
        """
        # validate input
        inputs_shape = inputs_l.size()  # store the original shape of the input
        targets = targets.squeeze() if targets.dim() > 1 else targets
        inputs_l = inputs_l.squeeze() if inputs_l.dim() > 2 else inputs_l
        inputs_u = inputs_u.squeeze() if inputs_u.dim() > 2 else inputs_u
        assert inputs_l.dim() == 2, f"Expected shape of (batchsize, num_classes), got {inputs_l.shape}"
        assert inputs_u.dim() == 2, f"Expected shape of (batchsize, num_classes), got {inputs_u.shape}"
        assert targets.dim() == 1, f"Expected shape of (batchsize,), got {targets.shape}"
        if targets.max() >= inputs_l.size(1):
            raise ValueError("Labels must be in the range of the output logit dimension.")

        # calculate the gradient bounds
        softmax_out_l, softmax_out_u = interval_arithmetic.propagate_softmax(inputs_l, inputs_u)
        n_classes = inputs_l.size(1)
        one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=n_classes)
        one_hot_labels = one_hot_labels.to(inputs_l.device).to(inputs_l.dtype)
        assert softmax_out_l.shape == one_hot_labels.shape, f"Expected {one_hot_labels.shape} got {softmax_out_l.shape}"

        if label_k_poison == 0:  # no label poisoning
            grad_l = (softmax_out_l - one_hot_labels).reshape(inputs_shape)
            grad_u = (softmax_out_u - one_hot_labels).reshape(inputs_shape)
        elif poison_target_idx == -1:  # untargeted label poisoning
            grad_l = (softmax_out_l - 1.0).reshape(inputs_shape)
            grad_u = (softmax_out_u - 0.0).reshape(inputs_shape)
        else:  # labels can only be flipped to the index specified by poison_target_idx
            one_hot_labels[:, poison_target_idx] = 1  # set the labels corresponding to the targeted index to 1
            grad_l = (softmax_out_l - one_hot_labels).reshape(inputs_shape)
            grad_u = (softmax_out_u - 0.0).reshape(inputs_shape)

        interval_arithmetic.validate_interval(grad_l, grad_u, msg="loss grad bounds")
        if self.reduction == "mean":
            LOGGER.warning("AGT processes grads in fragments, so using 'reduction=mean' is not recommended.")
            LOGGER.warning("Specify 'reduction=none' and normalise manually to ensure correct scaling.")
            return grad_l * n_classes / grad_l.nelement(), grad_u * n_classes / grad_u.nelement()
        return grad_l, grad_u


if __name__ == "__main__":
    # test that the loss function matches the standard BCE
    nominal_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = BoundedCrossEntropyLoss(reduction="mean")

    inputs = torch.randn(10, 5, requires_grad=True)
    targets = torch.randint(0, 5, size=(10,))
    nominal_loss_val = nominal_loss(inputs, targets)
    loss_val = loss.forward(inputs, targets)
    assert torch.allclose(nominal_loss_val, loss_val), f"Expected {nominal_loss_val}, got {loss_val}"
    nominal_loss_grad = torch.autograd.grad(loss_val.sum(), inputs)[0]
    loss_grad = loss.backward(inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad), f"Expected {nominal_loss_grad}, got {loss_grad}"

    # test that the bounded pass with zero input interval is the same as the unbounded pass
    loss_val_l, loss_val_u = loss.bound_forward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_val, loss_val_l)
    assert torch.allclose(nominal_loss_val, loss_val_u)
    loss_grad_l, loss_grad_u = loss.bound_backward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad_l)
    assert torch.allclose(nominal_loss_grad, loss_grad_u)

    loss_grad_l, loss_grad_u = loss.bound_backward(inputs, inputs, targets, label_k_poison=10)
    assert torch.all(loss_grad_l <= loss_grad_u), f"Got {loss_grad_l} and {loss_grad_u}"
    assert torch.all(loss_grad_l <= loss_grad), f"Got {loss_grad_l} and {loss_grad}"
    assert torch.all(loss_grad <= loss_grad_u + 1e-6), f"Got {loss_grad} and {loss_grad_u}"
