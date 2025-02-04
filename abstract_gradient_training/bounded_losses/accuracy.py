"""
Bounded version of the classification accuracy loss function. This loss is not differentiable, so does not implement the
backwards passes.
"""

import logging
from typing import Literal

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_losses import BoundedLoss

LOGGER = logging.getLogger(__name__)


class BoundedAccuracy(BoundedLoss):
    """
    Bounded classification accuracy loss function.
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
        target_shape = targets.size()
        targets = targets.squeeze() if targets.dim() > 1 else targets
        inputs = inputs.squeeze() if inputs.dim() > 2 else inputs

        # calculate predictions
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.size(-1) == 1):
            try:
                inputs = inputs.reshape(targets.size())  # binary classification
            except RuntimeError as e:
                raise ValueError(f"Logit shape {inputs.shape} does not match target shape {targets.shape}.") from e
            preds = inputs > 0
        elif inputs.dim() == 2:  # multi-class classification
            preds = torch.nn.functional.softmax(inputs, dim=1).argmax(dim=1)
        else:
            raise ValueError(
                f"Expected inputs to have shape (batchsize,) or (batchsize, num_classes), got {inputs.shape}"
            )

        # return accuracy
        preds = preds.reshape(targets.size())
        if self.reduction == "mean":
            return (preds == targets).float().mean()
        if self.reduction == "sum":
            return (preds == targets).float().sum()
        return (preds == targets).reshape(target_shape)

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
        Bounded forward pass through the loss function. Bounding poisoning attacks is not supported for this loss.

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Not supported for accuracy loss.
            label_epsilon (float, optional): Not supported for accuracy loss.
            poison_target_idx (int, optional): Not supported for accuracy loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the loss values.
        """
        if label_k_poison > 0 or label_epsilon > 0 or poison_target_idx >= 0:
            raise ValueError("Bounded accuracy loss does not support poisoning attacks.")

        # handle shapes and reshaping
        interval_arithmetic.validate_interval(inputs_l, inputs_u, msg="input bounds")
        target_shape = inputs_l.size()
        targets = targets.squeeze() if targets.dim() > 1 else targets
        inputs_l = inputs_l.squeeze() if inputs_l.dim() > 2 else inputs_l
        inputs_u = inputs_u.squeeze() if inputs_u.dim() > 2 else inputs_u
        # calculate predictions
        if inputs_l.dim() == 1 or (inputs_l.dim() == 2 and inputs_l.size(-1) == 1):  # binary classification
            try:
                inputs_l = inputs_l.reshape(targets.size())  # binary classification
                inputs_u = inputs_u.reshape(targets.size())  # binary classification
            except RuntimeError as e:
                raise ValueError(f"Logit shape {inputs_l.shape} does not match target shape {targets.shape}.") from e
            # worst case is minimizing the logits of points from class 1 and maximizing the logits of class 0
            worst_case_logits = (1 - targets) * inputs_u + targets * inputs_l
            worst_case_preds = worst_case_logits > 0
            best_case_logits = targets * inputs_u + (1 - targets) * inputs_l
            best_case_preds = best_case_logits > 0
        elif inputs_l.dim() == 2:  # multi-class classification
            assert targets.max() < inputs_l.size(1), "Labels must be in the range of the output logit dimension."
            # calculate best and worst case logits given the bounds
            one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=inputs_l.size(1)).to(inputs_l.device)
            worst_case_logits = (1 - one_hot_labels) * inputs_u + one_hot_labels * inputs_l
            best_case_logits = one_hot_labels * inputs_u + (1 - one_hot_labels) * inputs_l
            # calculate post-softmax output
            worst_case_preds = torch.nn.functional.softmax(worst_case_logits, dim=1).argmax(dim=1)
            best_case_preds = torch.nn.functional.softmax(best_case_logits, dim=1).argmax(dim=1)
        else:
            raise ValueError(
                f"Expected inputs to have shape (batchsize,) or (batchsize, num_classes), got {inputs_l.shape}"
            )

        # return accuracy
        worst_case_preds = worst_case_preds.reshape(targets.size())
        best_case_preds = best_case_preds.reshape(targets.size())
        if self.reduction == "mean":
            return (worst_case_preds == targets).float().mean(), (best_case_preds == targets).float().mean()
        if self.reduction == "sum":
            return (worst_case_preds == targets).float().sum(), (best_case_preds == targets).float().sum()
        return (worst_case_preds == targets).reshape(target_shape), (best_case_preds == targets).reshape(target_shape)

    def backward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients of the loss function with respect to the inputs to the loss function.
        """
        raise NotImplementedError("Accuracy loss is not differentiable.")

    def bound_backward(
        self, inputs_l: torch.Tensor, inputs_u: torch.Tensor, targets: torch.Tensor, **poison_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounds on the gradients of the loss function with respect to the inputs to the loss function.
        """
        raise NotImplementedError("Accuracy loss is not differentiable.")


if __name__ == "__main__":
    # test that the loss function matches the standard BCE
    loss = BoundedAccuracy(reduction="none")
    inputs = torch.randn(10, 1)
    targets = torch.randint(0, 2, (10, 1))
    assert (((inputs > 0) == targets) == loss.forward(inputs, targets)).all()

    inputs = torch.randn(10, 4)
    targets = torch.randint(0, 4, (10, 1))
    preds = torch.nn.functional.softmax(inputs, dim=1).argmax(dim=1, keepdim=True)
    assert (preds == targets).sum() == loss.forward(inputs, targets).sum()
