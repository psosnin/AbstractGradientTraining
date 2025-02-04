"""
Bounded version of the torch.optim.SGD optimizer with support for momentum.
"""

from __future__ import annotations

import torch

from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.bounded_optimizers import BoundedSGD


class BoundedSGDM(BoundedSGD):
    """
    A class implementing the SGD with momentum update step with optional learning rate decay. The update step applies to
    the parameters of the BoundedModel passed in at initialization. The optimizer applies the update to the parameter
    bounds in-place.

    The BoundedSGD update should be functionally identical to the torch.optim.SGD optimizer. However, the BoundedSGD
    also provides fairly aggressive learning rate decay (though this is off by default).
    """

    def __init__(self, bounded_model: BoundedModel, **optimizer_kwargs) -> None:
        # momentum parameters
        self.momentum = optimizer_kwargs.pop("momentum", 0.0)
        self.dampening = optimizer_kwargs.pop("dampening", 0.0)
        self.nesterov = optimizer_kwargs.pop("nesterov", False)
        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be between 0 and 1.")
        if not 0 <= self.dampening < 1:
            raise ValueError("Dampening must be between 0 and 1.")
        if self.nesterov and (self.momentum <= 0 or self.dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(bounded_model, **optimizer_kwargs)
        if self.l1_reg > 0:
            raise ValueError("L1 regularization is not supported with momentum.")
        self.l2_reg, self.l2_reg_momentum = 0.0, self.l2_reg  # we'll disable the l2 reg of the superclass

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
        # apply l2 regularization
        if self.l2_reg_momentum > 0:
            mean_grads_l = [g + self.l2_reg_momentum * p for g, p in zip(mean_grads_l, self.param_u)]
            mean_grads_n = [g + self.l2_reg_momentum * p for g, p in zip(mean_grads_n, self.param_n)]
            mean_grads_u = [g + self.l2_reg_momentum * p for g, p in zip(mean_grads_u, self.param_l)]

        # apply momentum
        if self.momentum != 0:
            # update "velocity" terms
            if self.step_count == 0:
                self.vel_l = mean_grads_l
                self.vel_n = mean_grads_n
                self.vel_u = mean_grads_u
            else:
                self.vel_l = [self.momentum * v + (1 - self.dampening) * g for v, g in zip(self.vel_l, mean_grads_l)]
                self.vel_n = [self.momentum * v + (1 - self.dampening) * g for v, g in zip(self.vel_n, mean_grads_n)]
                self.vel_u = [self.momentum * v + (1 - self.dampening) * g for v, g in zip(self.vel_u, mean_grads_u)]
            # add to the momentum
            if self.nesterov:
                mean_grads_l = [g + self.momentum * v for g, v in zip(mean_grads_l, self.vel_l)]
                mean_grads_n = [g + self.momentum * v for g, v in zip(mean_grads_n, self.vel_n)]
                mean_grads_u = [g + self.momentum * v for g, v in zip(mean_grads_u, self.vel_u)]
            else:
                mean_grads_l = self.vel_l
                mean_grads_n = self.vel_n
                mean_grads_u = self.vel_u

        # apply the update
        super().step(mean_grads_l, mean_grads_n, mean_grads_u)


if __name__ == "__main__":

    from abstract_gradient_training.bounded_losses import BoundedMSELoss
    from abstract_gradient_training.bounded_models import IntervalBoundedModel

    torch.manual_seed(0)
    test_model = torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.ReLU())
    bounded_model = IntervalBoundedModel(test_model)

    l2_reg = 0.1
    l1_reg = 0.0
    batchsize = 5
    momentum = 0.9
    nesterov = False
    n_iters = 10

    optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01, momentum=0.9, nesterov=nesterov)  # type: ignore
    batch, targets = torch.randn(batchsize, 10), torch.randn(batchsize, 2)

    for _ in range(n_iters):
        optimizer.zero_grad()
        out = test_model(batch)
        loss = torch.nn.functional.mse_loss(out, targets, reduction="sum") / batchsize
        loss += sum(l2_reg * p.norm(2) ** 2 / 2 for p in test_model.parameters())  # l2 reg
        loss += sum(l1_reg * p.norm(1) for p in test_model.parameters())  # l1 reg
        loss.backward()
        optimizer.step()

    print(test_model(batch))

    bounded_optimizer = BoundedSGDM(
        bounded_model, learning_rate=0.01, l2_reg=l2_reg, l1_reg=l1_reg, momentum=0.9, nesterov=nesterov
    )
    bounded_loss = BoundedMSELoss(reduction="sum")

    for _ in range(n_iters):
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
