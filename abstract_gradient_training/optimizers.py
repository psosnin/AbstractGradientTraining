"""Optimizer used by certified training to bound the model updates."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from abstract_gradient_training import interval_arithmetic

if TYPE_CHECKING:
    from abstract_gradient_training import AGTConfig


class SGD:
    """
    A class implementing the SGD update step with optional learning rate decay.
    """

    def __init__(self, config: AGTConfig) -> None:
        self.lr = config.learning_rate
        # get regularisation parameters
        self.l1_reg = config.l1_reg
        self.l2_reg = config.l2_reg
        # If these parameters are left to default, the optimizer will behave like a standard SGD with constant
        # learning rate. If you set these parameters, then the learning rate will decay like
        # lr = max(lr / (1 + sqrt(decay_rate * epoch), lr_min)
        self.lr_decay = config.lr_decay
        self.lr_min = config.lr_min
        self.epoch = 0

    def step(
        self,
        param_n: list[torch.Tensor],
        param_l: list[torch.Tensor],
        param_u: list[torch.Tensor],
        update_n: list[torch.Tensor],
        update_l: list[torch.Tensor],
        update_u: list[torch.Tensor],
        sound: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute a sound bound on parameters after an SGD update
            param_n = param_n - learning_rate * update_n.

        Args:
            param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
            param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
            param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
            update_n (list[torch.Tensor]): List of the nominal updates of the network [dW1, db1, ..., dWn, dbn].
            update_l (list[torch.Tensor]): List of the lower bound updates of the network [dW1, db1, ..., dWn, dbn].
            update_u (list[torch.Tensor]): List of the upper bound updates of the network [dW1, db1, ..., dWn, dbn].
            sound (bool, optional): Whether to check if the new param_n falls within the new bounds. Defaults to True.

        Returns:
            tuple: The updated parameter lists [param_n, param_l, param_u].
        """
        # apply regularisation
        param_n, param_l, param_u = l2_update(param_n, param_l, param_u, self.l2_reg, sound=sound)
        param_n, param_l, param_u = l1_update(param_n, param_l, param_u, self.l1_reg, sound=sound)
        lr = self.lr / (1 + self.lr_decay * self.epoch)
        lr = max(lr, self.lr_min)
        for i in range(len(param_n)):
            # apply the parameter update
            param_n[i] -= lr * update_n[i]
            param_l[i] -= lr * update_u[i]
            param_u[i] -= lr * update_l[i]
            if sound:
                interval_arithmetic.validate_interval(param_l[i], param_n[i])
                interval_arithmetic.validate_interval(param_n[i], param_u[i])
            else:
                interval_arithmetic.validate_interval(param_l[i], param_u[i])
        self.epoch += 1
        return param_n, param_l, param_u


def l1_update(
    param_n: list[torch.Tensor],
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    l1_reg: float,
    sound: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute a sound bound on the l1 regularisation parameter update
        param_n = param_n - l1_reg * torch.sign(param_n)
    using interval arithmetic.

    Args:
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        l1_reg (float): The l1 regularisation parameter.
        sound (bool, optional): Whether to check if the new param_n falls within the new bounds. Defaults to True.

    Returns:
        tuple: The updated parameter lists [param_n, param_l, param_u].
    """
    for i in range(len(param_n)):
        # compute L1 regularisation parameter update
        param_n[i] = param_n[i] - l1_reg * torch.sign(param_n[i])
        # handle edge case where the l1 update causes the bounds to cross zero
        # clamp crossing indices and update non crossing indices as normal
        crossing = (param_l[i] <= 0) & (param_u[i] >= 0)
        param_l[i] = torch.where(
            crossing, torch.clamp(param_l[i] + l1_reg, max=-l1_reg), param_l[i] - l1_reg * torch.sign(param_l[i])
        )
        param_u[i] = torch.where(
            crossing, torch.clamp(param_u[i] - l1_reg, min=l1_reg), param_u[i] - l1_reg * torch.sign(param_u[i])
        )
        if sound:
            interval_arithmetic.validate_interval(param_l[i], param_u[i])
            interval_arithmetic.validate_interval(param_n[i], param_u[i])
        else:
            interval_arithmetic.validate_interval(param_l[i], param_u[i])
    return param_n, param_l, param_u


def l2_update(
    param_n: list[torch.Tensor],
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    l2_reg: float,
    sound: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute a sound bound on the l2 regularisation parameter update using interval arithmetic.

    Args:
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        l1_reg (float): The l1 regularisation parameter.
        sound (bool, optional): Whether to check if the new param_n falls within the new bounds. Defaults to True.

    Returns:
        tuple: The updated parameter lists [param_n, param_l, param_u].
    """
    assert 0 <= l2_reg <= 1, "l2_reg must be in the range [0, 1]"
    for i in range(len(param_n)):
        param_n[i] = (1 - l2_reg) * param_n[i]
        param_l[i] = (1 - l2_reg) * param_l[i]
        param_u[i] = (1 - l2_reg) * param_u[i]
        if sound:
            interval_arithmetic.validate_interval(param_l[i], param_u[i])
            interval_arithmetic.validate_interval(param_n[i], param_u[i])
        else:
            interval_arithmetic.validate_interval(param_l[i], param_u[i])
    return param_n, param_l, param_u
