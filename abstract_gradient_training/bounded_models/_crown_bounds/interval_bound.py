"""A class representing an interval over a pytorch tensor."""

from __future__ import annotations
from typing import Literal
from collections.abc import Iterator

import torch

from abstract_gradient_training import interval_arithmetic


class IntervalBounds:
    """
    A class representing an interval over a pytorch tensor. This class overloads the basic arithmetic operations to
    support interval arithmetic.
    """

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor | None = None,
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
    ):
        """
        Initialise the interval tensor.

        Args:
            lb: The lower bound of the interval.
            ub: The upper bound of the interval. If ub is None, then we set self.lb = self.ub = lb.
            interval_matmul: The method to use for interval matrix multiplication. One of "rump", "exact", "nguyen".

        """
        assert isinstance(lb, torch.Tensor)
        assert isinstance(ub, torch.Tensor | None)
        if ub is not None:
            assert lb.shape == ub.shape
            interval_arithmetic.validate_interval(lb, ub)
        self.lb = lb
        self.ub = lb if ub is None else ub
        self.shape = lb.shape
        self.dtype = lb.dtype
        self.interval_matmul: Literal["rump", "exact", "nguyen"] = interval_matmul

    def __repr__(self):
        return f"IntervalBounds(\n{self.lb},\n{self.ub}\n)"

    def __str__(self):
        return f"IntervalBounds(\n{self.lb},\n{self.ub}\n)"

    def __add__(self, other: IntervalBounds | torch.Tensor) -> IntervalBounds:
        """Add two intervals or an interval plus a constant using A + B."""
        if isinstance(other, IntervalBounds):
            return IntervalBounds(self.lb + other.lb, self.ub + other.ub)
        return IntervalBounds(self.lb + other, self.ub + other)

    def __or__(self, other: IntervalBounds) -> IntervalBounds:
        """Union of two intervals using A | B."""
        return IntervalBounds(torch.minimum(self.lb, other.lb), torch.maximum(self.ub, other.ub))

    def __matmul__(self, other: IntervalBounds) -> IntervalBounds:
        """
        Matrix product of two intervals A @ B.
        """
        assert isinstance(other, IntervalBounds)
        return IntervalBounds(
            *interval_arithmetic.propagate_matmul(self.lb, self.ub, other.lb, other.ub, self.interval_matmul)
        )

    def __mul__(self, other: IntervalBounds | torch.Tensor) -> IntervalBounds:
        """Elementwise product of two intervals A * B."""
        if not isinstance(other, IntervalBounds):
            if (other >= 0).all():
                return IntervalBounds(self.lb * other, self.ub * other)
            cases = torch.stack([self.lb * other, self.ub * other])
        else:
            cases = torch.stack([self.lb * other.lb, self.ub * other.lb, self.lb * other.ub, self.ub * other.ub])
        lb = torch.min(cases, dim=0)[0]
        ub = torch.max(cases, dim=0)[0]
        return IntervalBounds(lb, ub)

    def __getitem__(self, key: int | slice | torch.Tensor) -> IntervalBounds:
        return IntervalBounds(self.lb[key], self.ub[key])

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Implement iter for unpacking."""
        return iter((self.lb, self.ub))

    def isin(self, x: torch.Tensor) -> bool:
        """Check if x is in the interval."""
        return bool((self.lb <= x).all() and (x <= self.ub).all())

    def get_device(self) -> int:
        """Apply pytorch get_device."""
        return self.lb.get_device()

    def sum(self, dimension) -> IntervalBounds:
        """Apply pytorch sum."""
        return IntervalBounds(self.lb.sum(dimension), self.ub.sum(dimension))

    def unsqueeze(self, dim=-1) -> IntervalBounds:
        """Apply pytorch unsqueeze."""
        return IntervalBounds(self.lb.unsqueeze(dim), self.ub.unsqueeze(dim))

    def squeeze(self, *args) -> IntervalBounds:
        """Apply pytorch squeeze."""
        return IntervalBounds(self.lb.squeeze(*args), self.ub.squeeze(*args))

    @property
    def mT(self) -> IntervalBounds:
        """Apply pytorch transpose to the last 2 dims."""
        return IntervalBounds(self.lb.mT, self.ub.mT)

    def transpose(self, dim1: int, dim2: int) -> IntervalBounds:
        """Apply pytorch transpose."""
        return IntervalBounds(self.lb.transpose(dim1, dim2), self.ub.transpose(dim1, dim2))

    def dim(self) -> int:
        """Return the number of dimensions of the interval."""
        return self.lb.dim()

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the lower and upper bounds as a tuple."""
        return self.lb, self.ub

    def clone(self) -> IntervalBounds:
        """Return a copy of the interval tensor."""
        return IntervalBounds(self.lb.clone(), self.ub.clone())

    @staticmethod
    def zeros_like(
        A: IntervalBounds | torch.Tensor,
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
    ) -> IntervalBounds:
        """Create a zero interval with the same shape as A."""
        if isinstance(A, IntervalBounds):
            z = torch.zeros_like(A.lb).type(A.dtype)
        else:
            z = torch.zeros_like(A).type(A.dtype)
        return IntervalBounds(z, z, interval_matmul=interval_matmul)

    @staticmethod
    def zeros(*args, interval_matmul: Literal["rump", "exact", "nguyen"] = "rump", **kwargs) -> IntervalBounds:
        """Create a zero interval with the given shape."""
        z = torch.zeros(*args, **kwargs)
        return IntervalBounds(z, z, interval_matmul=interval_matmul)
