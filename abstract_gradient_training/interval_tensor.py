from typing import Optional
import torch
import torch.nn.functional as F

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training import bound_utils


class IntervalTensor:
    """
    A class representing an interval over a pytorch tensor. This class overloads the basic arithmetic operations to
    support interval arithmetic.
    """

    def __init__(self, lb, ub=None):
        """
        If ub is None, then the interval is constant and we set self.lb = self.ub = lb.
        """
        assert isinstance(lb, torch.Tensor)
        assert isinstance(ub, Optional[torch.Tensor])
        if ub is not None:
            assert lb.shape == ub.shape
            bound_utils.validate_interval(lb, ub)
        self.lb = lb
        self.ub = lb if ub is None else ub
        self.shape = lb.shape
        self.dtype = lb.dtype
        self.tuple = (self.lb, self.ub)

    def __repr__(self):
        return f"IntervalTensor(\n{self.lb},\n{self.ub}\n)"

    def __str__(self):
        return f"IntervalTensor(\n{self.lb},\n{self.ub}\n)"

    def __add__(self, other):
        """
        Add two intervals or an interval plus a constant using A + B.
        """
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.lb + other.lb, self.ub + other.ub)
        return IntervalTensor(self.lb + other, self.ub + other)

    def __or__(self, other):
        """
        Union of two intervals using A | B
        """
        return IntervalTensor(torch.minimum(self.lb, other.lb), torch.maximum(self.ub, other.ub))

    def __matmul__(self, other):
        """
        Matrix product of two intervals A @ B.
        NOTE: If B is a constant it must be elementwise >= 0 for the returned interval to be valid.
        """
        assert isinstance(other, IntervalTensor)
        return IntervalTensor(*interval_arithmetic.propagate_matmul(self.lb, self.ub, other.lb, other.ub))

    def __mul__(self, other):
        """
        Elementwise product of two intervals A * B.
        """
        if isinstance(other, tuple):
            other = IntervalTensor(*other)
        if not isinstance(other, IntervalTensor):
            if (other >= 0).all():
                return IntervalTensor(self.lb * other, self.ub * other)
            else:
                cases = torch.stack([self.lb * other, self.ub * other])
        else:
            cases = torch.stack([self.lb * other.lb, self.ub * other.lb, self.lb * other.ub, self.ub * other.ub])
        lb = torch.min(cases, dim=0)[0]
        ub = torch.max(cases, dim=0)[0]
        return IntervalTensor(lb, ub)

    """ Misc helper functions. """

    def isin(self, x):
        return (self.lb <= x).all() and (x <= self.ub).all()

    def heaviside(self):
        return IntervalTensor((self.lb > 0).type(self.dtype), (self.ub > 0).type(self.dtype))

    def relu(self):
        return IntervalTensor(F.relu(self.lb), F.relu(self.ub))

    def radius_norm(self):
        return (self.ub - self.lb).norm()

    """ Overloading of misc pytorch methods """

    def __getitem__(self, key):
        return IntervalTensor(self.lb[key], self.ub[key])

    def get_device(self):
        return self.lb.get_device()

    def sum(self, dimension):
        return IntervalTensor(self.lb.sum(dimension), self.ub.sum(dimension))

    def unsqueeze(self, dim=-1):
        return IntervalTensor(self.lb.unsqueeze(dim), self.ub.unsqueeze(dim))

    def T(self):
        return IntervalTensor(self.lb.transpose(-2, -1), self.ub.transpose(-2, -1))

    def transpose(self, dim1, dim2):
        return IntervalTensor(self.lb.transpose(dim1, dim2), self.ub.transpose(dim1, dim2))

    """ Static methods for creating intervals. """

    @staticmethod
    def zeros_like(A):
        """
        Create a zero interval with the same shape as A.
        """
        if isinstance(A, IntervalTensor):
            z = torch.zeros_like(A.lb).type(A.dtype)
        else:
            z = torch.zeros_like(A).type(A.dtype)
        return IntervalTensor(z, z)

    @staticmethod
    def zeros(*args, **kwargs):
        """
        Create a zero interval with the given shape.
        """
        z = torch.zeros(*args, **kwargs)
        return IntervalTensor(z, z)
