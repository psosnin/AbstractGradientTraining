"""Abstract base class for bounded versions of pytorch loss functions."""

import abc

import torch


class BoundedLoss(abc.ABC):
    """
    Abstract base class for bounded versions of pytorch loss functions which implement the following operations:

    - forward: Nominal forward pass of the loss function.
    - bound_forward: Bounded forward pass of the loss function.
    - backward: Nominal backward pass of the loss function (computes gradients of the loss wrt inputs)
    - bound_backward: Bounded backward pass of the loss function (computes gradients of the loss wrt inputs)
    """

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Nominal forward pass through the loss function.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss values.
        """

    @abc.abstractmethod
    def backward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Return the gradients of the loss wrt inputs.

        Args:
            inputs (torch.Tensor): Inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.

        Returns:
            torch.Tensor: Loss gradient values.
        """

    @abc.abstractmethod
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
            - label_epsilon: (Regression only) Maximum perturbation of the targets (in the inf norm).
            - poison_target_idx (Classification only) Target class for the poisoning attack. If -1, then the attacker
                may flip labels to any class. Otherwise, the attacker may only flip labels to the class with the
                specified index.

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Maximum number of data-points with poisoned targets.
            label_epsilon (float, optional): Maximum perturbation of the targets (in the inf norm).
            poison_target_idx (int, optional): Target class for the poisoning attack.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the loss values.
        """

    @abc.abstractmethod
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
            - label_epsilon: (Regression only) Maximum perturbation of the targets (in the inf norm).
            - poison_target_idx (Classification only) Target class for the poisoning attack. If -1, then the attacker
                may flip labels to any class. Otherwise, the attacker may only flip labels to the class with the
                specified index.

        Args:
            inputs_l (torch.Tensor): Lower bounds on the inputs to the loss function.
            inputs_u (torch.Tensor): Upper bounds on the inputs to the loss function.
            target (torch.Tensor): Target values for the loss function.
            label_k_poison (int, optional): Maximum number of data-points with poisoned targets.
            label_epsilon (float, optional): Maximum perturbation of the targets (in the inf norm).
            poison_target_idx (int, optional): Target class for the poisoning attack.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the gradients of the loss.
        """
