"""Bounded version of the torch.nn.BCEWithLogitsLoss loss function."""

import logging
from typing import Literal

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_losses import BoundedLoss

LOGGER = logging.getLogger(__name__)


class BoundedBCEWithLogitsLoss(BoundedLoss):
    """
    Bounded version of the torch.nn.BCEWithLogitsLoss() loss function.
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
        if not ((targets == 0) + (targets == 1)).all():
            raise ValueError("Labels must be binary for binary cross-entropy loss.")
        # handle shapes and types
        input_shape = inputs.shape  # save the input shape, in case we need to return the gradient
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs.dtype)
        try:
            inputs = inputs.reshape(targets.shape)
        except RuntimeError as e:
            raise RuntimeError(f"Logits shape {inputs.shape} does not match targets {targets.shape} for BCE.") from e

        # compute the loss
        outputs = torch.sigmoid(inputs)
        bce_val = -(targets * torch.log(outputs + 1e-10) + (1 - targets) * torch.log(1 - outputs + 1e-10))

        # perform the reduction over any remaining dimensions
        if self.reduction == "sum":
            return torch.sum(bce_val)
        elif self.reduction == "mean":
            return torch.mean(bce_val)
        return bce_val.reshape(input_shape)

    def backward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Return the gradients of the loss wrt inputs.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss gradient values.
        """
        if not ((targets == 0) + (targets == 1)).all():
            raise ValueError("Labels must be binary for binary cross-entropy loss.")
        # handle shapes and types
        input_shape = inputs.shape  # save the input shape, in case we need to return the gradient
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs.dtype)
        try:
            inputs = inputs.reshape(targets.shape)
        except RuntimeError as e:
            raise RuntimeError(f"Logits shape {inputs.shape} does not match targets {targets.shape} for BCE.") from e

        # compute the loss gradient
        grad = (torch.sigmoid(inputs) - targets).reshape(input_shape)
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
        Bounded forward pass through the loss function. The forward pass of this loss does not support label poisoning.

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Ignored for the BCE forward pass.
            label_epsilon (float, optional): Ignored for the BCE forward pass.
            poison_target_idx (int, optional): Ignored for the BCE forward pass.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the loss values.
        """
        # validate input
        if label_k_poison > 0 or label_epsilon > 0 or poison_target_idx >= 0:
            raise ValueError("BCE loss does not support label poisoning for the forward pass.")
        if not ((targets == 0) + (targets == 1)).all():
            raise ValueError("Labels must be binary for binary cross-entropy loss.")
        interval_arithmetic.validate_interval(inputs_l, inputs_u, msg="loss input bounds")

        # handle shapes and types
        input_shape = inputs_l.shape  # save the input shape, in case we need to return the gradient
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs_l.dtype)
        try:
            inputs_l, inputs_u = inputs_l.reshape(targets.shape), inputs_u.reshape(targets.shape)
        except RuntimeError as e:
            raise RuntimeError(f"Logits shape {inputs_l.shape} does not match targets {targets.shape} for BCE.") from e

        # un-poisoned case: compute the best and worst case outputs by min/maxing the logits corresponding to the labels
        worst_outputs = torch.sigmoid((1 - targets) * inputs_u + targets * inputs_l)
        best_outputs = torch.sigmoid(targets * inputs_u + (1 - targets) * inputs_l)
        # lowest loss computed with best logits, highest loss computed with worst logits
        bce_val_l = -(targets * torch.log(best_outputs + 1e-8) + (1 - targets) * torch.log(1 - best_outputs + 1e-10))
        bce_val_u = -(targets * torch.log(worst_outputs + 1e-10) + (1 - targets) * torch.log(1 - worst_outputs + 1e-10))
        interval_arithmetic.validate_interval(bce_val_l, bce_val_u, msg="loss bounds")

        # perform the reduction over any remaining dimensions
        if self.reduction == "sum":
            return torch.sum(bce_val_l), torch.sum(bce_val_u)
        elif self.reduction == "mean":
            return torch.mean(bce_val_l), torch.mean(bce_val_u)
        return bce_val_l.reshape(input_shape), bce_val_u.reshape(input_shape)

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

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Maximum number of data-points with poisoned targets.
            label_epsilon (float, optional): Not supported for BCE loss.
            poison_target_idx (int, optional): Not supported for BCE loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the gradients of the loss.
        """
        if poison_target_idx >= 0:
            raise ValueError("BCE loss does not support targeted label poisoning.")
        if label_epsilon > 0:
            raise ValueError("BCE loss does not support (inf-norm) perturbation to the targets.")

        # handle shapes and types
        input_shape = inputs_l.shape  # save the input shape, in case we need to return the gradient
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs_l.dtype)
        try:
            inputs_l, inputs_u = inputs_l.reshape(targets.shape), inputs_u.reshape(targets.shape)
        except RuntimeError as e:
            raise RuntimeError(f"Logits shape {inputs_l.shape} does not match targets {targets.shape} for BCE.") from e
        interval_arithmetic.validate_interval(inputs_l, inputs_u, msg="loss input bounds")

        if label_k_poison != 0:  # label flipping case
            grad_l = (torch.sigmoid(inputs_l) - 1).reshape(input_shape)
            grad_u = (torch.sigmoid(inputs_u) - 0).reshape(input_shape)
        else:  # un-poisoned labels
            grad_l = (torch.sigmoid(inputs_l) - targets).reshape(input_shape)
            grad_u = (torch.sigmoid(inputs_u) - targets).reshape(input_shape)
        interval_arithmetic.validate_interval(grad_l, grad_u, msg="loss grad bounds")

        if self.reduction == "mean":
            LOGGER.warning("AGT processes grads in fragments, so using 'reduction=mean' is not recommended.")
            LOGGER.warning("Specify 'reduction=none' and normalise manually to ensure correct scaling.")
            return grad_l / grad_l.nelement(), grad_u / grad_u.nelement()
        return grad_l, grad_u


if __name__ == "__main__":
    # test that the loss function matches the standard BCE
    nominal_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = BoundedBCEWithLogitsLoss(reduction="mean")

    # validate forward passes
    inputs = torch.randn(10, 1, requires_grad=True)
    targets = torch.randint(0, 2, size=(10, 1))
    nominal_loss_val = nominal_loss(inputs, targets.float())
    loss_val = loss.forward(inputs, targets)
    loss_val_l, loss_val_u = loss.bound_forward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_val, loss_val), f"Expected {nominal_loss_val}, got {loss_val}"
    assert torch.allclose(nominal_loss_val, loss_val_l)
    assert torch.allclose(nominal_loss_val, loss_val_u)

    # validate backward passes
    nominal_loss_grad = torch.autograd.grad(loss_val.sum(), inputs, create_graph=True)[0]
    loss_grad = loss.backward(inputs, targets)
    loss_grad_l, loss_grad_u = loss.bound_backward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad), f"Expected {nominal_loss_grad}, got {loss_grad}"
    assert torch.allclose(nominal_loss_grad, loss_grad_l)
    assert torch.allclose(nominal_loss_grad, loss_grad_u)
