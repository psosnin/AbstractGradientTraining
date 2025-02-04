"""
Bounded version of the torch.optim.SGD optimizer.
"""

from __future__ import annotations

import logging

import torch

from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.bounded_optimizers import BoundedOptimizer
from abstract_gradient_training import interval_arithmetic

LOGGER = logging.getLogger(__name__)


class BoundedSGD(BoundedOptimizer):
    """
    A class implementing the SGD update step with optional learning rate decay. The update step applies to the
    parameters of the BoundedModel passed in at initialization. The optimizer applies the update to the parameter bounds
    in-place.

    The BoundedSGD update should be functionally identical to the torch.optim.SGD optimizer. However, the BoundedSGD
    also provides fairly aggressive learning rate decay (though this is off by default).

    """

    def __init__(self, bounded_model: BoundedModel, **optimizer_kwargs) -> None:
        # store config values
        self.lr = optimizer_kwargs.pop("learning_rate", -1)
        self.l1_reg = optimizer_kwargs.pop("l1_reg", 0.0)
        self.l2_reg = optimizer_kwargs.pop("l2_reg", 0.0)
        # validate the parameters
        if self.lr <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 <= self.l1_reg:
            raise ValueError("l1_reg must be non-negative.")
        if not 0 <= self.l2_reg:
            raise ValueError("l2_reg must be non-negative.")
        if self.l1_reg > 0 and self.l2_reg > 0:
            raise ValueError("l1_reg and l2_reg cannot be used together, the bounds interact in a non-trivial way.")

        # store the parameters and bounds
        self.param_l = bounded_model.param_l
        self.param_n = bounded_model.param_n
        self.param_u = bounded_model.param_u

        # Initialise the learning rate scheduler. If the following parameters are left to default, the optimizer will
        # behave like a standard SGD with constant learning rate.
        lr_decay = optimizer_kwargs.pop("lr_decay", 0.0)
        lr_min = optimizer_kwargs.pop("lr_min", 0.0)
        self.lr_scheduler = lambda step: max(self.lr / (1 + lr_decay * step), lr_min)

        if lr_decay < 0:
            raise ValueError("lr_decay must be non-negative.")
        if lr_min < 0:
            raise ValueError("lr_min must be non-negative.")
        self.step_count = 0

        if optimizer_kwargs:
            LOGGER.warning(f"Unrecognized optimizer_kwargs: {optimizer_kwargs.keys()}")

    @torch.no_grad()
    def step(
        self, mean_grads_l: list[torch.Tensor], mean_grads_n: list[torch.Tensor], mean_grads_u: list[torch.Tensor]
    ):
        """
        Apply a sound SGD update the parameters and their bounds.

        Args:
            mean_grads_l (list[torch.Tensor]): Lower bound on the mean gradient of the batch.
            mean_grads_n (list[torch.Tensor]): Nominal mean gradient of the batch.
            mean_grads_u (list[torch.Tensor]): Upper bound on the mean gradient of the batch.

        Returns:
            tuple: The updated parameter lists [param_l, param_n, param_u].
        """
        # compute the new learning rate
        lr = self.lr_scheduler(self.step_count)

        # validate the gradient interval and shapes
        for i in range(len(mean_grads_n)):
            if mean_grads_n[i].shape != self.param_n[i].shape:
                raise ValueError(
                    f"Gradient shape mismatch. Expected {self.param_n[i].shape}, got {mean_grads_n[i].shape}."
                )
            interval_arithmetic.validate_interval(mean_grads_l[i], mean_grads_u[i], mean_grads_n[i], msg="pre sgd")

        # apply l2 regularization
        if self.l2_reg > 0:
            mean_grads_l = [g + self.l2_reg * p for g, p in zip(mean_grads_l, self.param_u)]
            mean_grads_n = [g + self.l2_reg * p for g, p in zip(mean_grads_n, self.param_n)]
            mean_grads_u = [g + self.l2_reg * p for g, p in zip(mean_grads_u, self.param_l)]

        # apply l1 regularization
        if self.l1_reg > 0:
            self.l1_update()

        for i in range(len(mean_grads_n)):
            # apply the parameter update
            self.param_n[i] -= lr * mean_grads_n[i]
            self.param_l[i] -= lr * mean_grads_u[i]
            self.param_u[i] -= lr * mean_grads_l[i]
            interval_arithmetic.validate_interval(self.param_l[i], self.param_u[i], self.param_n[i], msg="post sgd")

        self.step_count += 1

    @torch.no_grad()
    def l1_update(self) -> None:
        """
        Compute a sound bound on the l1 regularisation parameter update
            param_n = param_n - l1_reg * torch.sign(param_n)
        using interval arithmetic.
        """
        lr = self.lr_scheduler(self.step_count)
        for pl, pn, pu in zip(self.param_l, self.param_n, self.param_u):
            interval_arithmetic.validate_interval(pl, pu, pn, msg="pre l1 reg")
            l1_scale = lr * self.l1_reg
            pn -= l1_scale * torch.sign(pn)  # nominal l1 update
            # for the bounds, we need to handle the case where the bounds cross zero. we'll do this by clamping the
            # zero-crossing indices and update non crossing indices as normal
            crossing_idx = (pl <= 0) & (pu >= 0)  # zero-crossing indices
            # these have to be in-place
            pl.data.copy_(torch.clamp(pl + l1_scale, max=-l1_scale).where(crossing_idx, pl - l1_scale * torch.sign(pl)))
            pu.data.copy_(torch.clamp(pu - l1_scale, min=l1_scale).where(crossing_idx, pu - l1_scale * torch.sign(pu)))
            interval_arithmetic.validate_interval(pl, pu, pn, msg="post l1 reg")


if __name__ == "__main__":

    from abstract_gradient_training.bounded_losses import BoundedMSELoss
    from abstract_gradient_training.bounded_models import IntervalBoundedModel

    torch.manual_seed(0)
    test_model = torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.ReLU())
    bounded_model = IntervalBoundedModel(test_model)

    l2_reg = 0.1
    l1_reg = 1.0
    batchsize = 5

    optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)  # type: ignore
    batch, targets = torch.randn(batchsize, 10), torch.randn(batchsize, 2)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets, reduction="sum") / batchsize
    loss += sum(l2_reg * p.norm(2) ** 2 / 2 for p in test_model.parameters())  # l2 reg
    loss += sum(l1_reg * p.norm(1) for p in test_model.parameters())  # l1 reg

    loss.backward()
    optimizer.step()
    print(test_model(batch))

    bounded_optimizer = BoundedSGD(bounded_model, learning_rate=0.01, l2_reg=l2_reg, l1_reg=l1_reg)
    bounded_loss = BoundedMSELoss(reduction="sum")
    logit_n = bounded_model.forward(batch, retain_intermediate=True)
    loss_grad_n = bounded_loss.backward(logit_n, targets)
    grads_n = bounded_model.backward(loss_grad_n)

    logit_l, logit_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(logit_l, logit_u, targets)
    grads_l, grads_u = bounded_model.bound_backward(loss_grad_l, loss_grad_u)
    grads_l = [g.mean(dim=0) for g in grads_l]
    grads_u = [g.mean(dim=0) for g in grads_u]
    grads_n = [g.mean(dim=0) for g in grads_n]
    bounded_optimizer.step(grads_l, grads_n, grads_u)
    print(bounded_model.forward(batch))
