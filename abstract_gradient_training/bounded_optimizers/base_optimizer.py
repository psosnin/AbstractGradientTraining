"""Base class for optimizers used by certified training to bound the model updates."""

from __future__ import annotations
import abc

import torch

from abstract_gradient_training.bounded_models import BoundedModel


class BoundedOptimizer(abc.ABC):
    """
    A base class for the BoundedOptimizer interface used by certified training to bound the model updates.
    """

    @abc.abstractmethod
    def __init__(self, bounded_model: BoundedModel, **optimizer_kwargs) -> None:
        """
        Bind the parameters of the bounded_model to the optimizer

        Args:
            bounded_model (BoundedModel): Bounded model whose parameters will be updated by the optimizer.
            optimizer_kwargs (dict): Additional keyword arguments to be passed to the optimizer
        """

    @abc.abstractmethod
    def step(
        self, mean_grads_l: list[torch.Tensor], mean_grads_n: list[torch.Tensor], mean_grads_u: list[torch.Tensor]
    ) -> None:
        """
        Given the bounds and nominal value of the gradients in this batch, apply the optimizer update to the
        parameter values and bounds. The update should be applied in-place.

        Args:
            mean_grads_l (list[torch.Tensor]): Lower bound of the gradients (mean over batch dim).
            mean_grads_n (list[torch.Tensor]): Nominal value of the gradients (mean over batch dim).
            mean_grads_u (list[torch.Tensor]): Upper bound of the gradients (mean over batch dim).
        """
