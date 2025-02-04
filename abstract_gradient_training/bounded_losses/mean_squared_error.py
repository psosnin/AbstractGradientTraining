"""Bounded version of the mean squared error loss function."""

from typing import Literal
import logging

import torch
from abstract_gradient_training import interval_arithmetic

from abstract_gradient_training.bounded_losses import BoundedLoss

LOGGER = logging.getLogger(__name__)


class BoundedMSELoss(BoundedLoss):
    """
    Bounded version of the torch.nn.MSELoss() loss function.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        """
        Args:
            reduction (str, optional): Specifies the reduction to apply to the output.
        """
        if reduction not in ["sum", "mean", "none"]:
            raise ValueError(f"Reduction {reduction} not recognised.")
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Nominal forward pass through the loss function.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss values.
        """
        targets = targets.reshape(inputs.shape)
        mse_val = torch.square(inputs - targets)
        if self.reduction == "sum":
            return torch.sum(mse_val)
        elif self.reduction == "mean":
            return torch.mean(mse_val)
        return mse_val

    def backward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Return the gradients of the loss wrt inputs.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss gradient values.
        """
        targets = targets.reshape(inputs.shape)
        grad = 2 * (inputs - targets)
        if self.reduction == "mean":
            LOGGER.warning("AGT processes grads in fragments, so using 'reduction=mean' is not recommended.")
            LOGGER.warning("Specify 'reduction=none' and normalise manually to ensure correct scaling.")
            return grad / grad.nelement()
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
        Bounded forward pass through the loss function. If a poisoning attack is specified, we also compute bounds wrt
        the poisoning attack with the following parameters:

            - label_k_poison: Maximum number of data-points with poisoned targets.
            - label_epsilon: Maximum perturbation of the targets (in the inf norm).

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Maximum number of data-points with poisoned targets.
            label_epsilon (float, optional): Maximum perturbation of the targets (in the inf norm).
            poison_target_idx (int, optional): Not supported for the MSE loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the loss values.
        """
        if poison_target_idx != -1:
            raise ValueError("MSE loss does not support the poison_target_idx parameter.")
        if not label_epsilon >= 0:
            raise ValueError("Target epsilon must be greater than 0.")
        if inputs_l.shape != inputs_u.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        label_epsilon = label_epsilon if label_k_poison > 0 else 0.0

        # check shapes
        targets = targets.reshape(inputs_l.shape)
        interval_arithmetic.validate_interval(inputs_l, inputs_u, msg="mse loss input bounds.")
        diffs_l = inputs_l - targets - label_epsilon
        diffs_u = inputs_u - targets + label_epsilon

        # calculate the mse lower bound with the following cases:
        # 1. if the differences span zero, then the best MSE is zero
        # 2. if the differences are positive, best case MSE is lb
        # 3. if the differences are negative, best case MSE is ub
        mse_l = torch.zeros_like(diffs_l) + (diffs_l**2) * (diffs_l > 0) + (diffs_u**2) * (diffs_u < 0)
        mse_u = torch.maximum(diffs_l**2, diffs_u**2)
        interval_arithmetic.validate_interval(mse_l, mse_u, msg="mse loss value bounds.")

        if self.reduction == "sum":
            return torch.sum(mse_l), torch.sum(mse_u)
        elif self.reduction == "mean":
            return torch.mean(mse_l), torch.mean(mse_u)
        return mse_l, mse_u

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
            - label_epsilon: Maximum perturbation of the targets (in the inf norm).

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Maximum number of data-points with poisoned targets.
            label_epsilon (float, optional): Maximum perturbation of the targets (in the inf norm).
            poison_target_idx (int, optional): Not supported for the MSE loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the gradients of the loss.
        """
        if poison_target_idx != -1:
            raise ValueError("MSE loss does not support the poison_target_idx parameter.")
        if not label_epsilon >= 0:
            raise ValueError("Target epsilon must be greater than 0.")
        if inputs_l.shape != inputs_u.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        label_epsilon = label_epsilon if label_k_poison > 0 else 0.0
        targets = targets.reshape(inputs_l.shape)
        interval_arithmetic.validate_interval(inputs_l, inputs_u, msg="mse loss input bounds.")
        grad_l = 2 * (inputs_l - targets - label_epsilon)
        grad_u = 2 * (inputs_u - targets + label_epsilon)
        if self.reduction == "mean":
            LOGGER.warning("AGT processes grads in fragments, so using 'reduction=mean' is not recommended.")
            LOGGER.warning("Specify 'reduction=none' and normalise manually to ensure correct scaling.")
            return grad_l / grad_l.nelement(), grad_u / grad_u.nelement()
        return grad_l, grad_u


if __name__ == "__main__":
    # test that the loss function matches the standard MSE loss
    nominal_loss = torch.nn.MSELoss(reduction="sum")
    loss = BoundedMSELoss(reduction="sum")

    # test with a multi-dimensional input, e.g. multi-dimensional regression
    inputs = torch.randn(10, 1, requires_grad=True)
    targets = torch.randn(10, 1)
    nominal_loss_val = nominal_loss(inputs, targets)
    loss_val = loss.forward(inputs, targets)
    assert torch.allclose(nominal_loss_val, loss_val)
    nominal_loss_grad = torch.autograd.grad(loss_val.sum(), inputs, create_graph=True)[0]
    loss_grad = loss.backward(inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad)

    # test that the bounded pass with zero input interval is the same as the unbounded pass
    loss_val_l, loss_val_u = loss.bound_forward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_val, loss_val_l)
    assert torch.allclose(nominal_loss_val, loss_val_u)
    loss_grad_l, loss_grad_u = loss.bound_backward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad_l)
    assert torch.allclose(nominal_loss_grad, loss_grad_u)
