"""Class representing 'simple' linear bounds on a variable consisting of a single upper and lower bound."""

from typing import Iterator

from abstract_gradient_training.bounded_models._crown_bounds import IntervalBounds


class LinearBounds:
    """
    Class representing linear bounds of the form
        Omega @ x + theta <= out_var <= Lambda @ x + delta
    where Omega and Lambda are interval matrices and theta and delta are interval vectors.

    The linear bounds must have the following shapes:
        Lambda: [batchsize x m x n]
        Omega: [batchsize x m x n]
        delta: [batchsize x m]
        theta: [batchsize x m]

    Note that even if batchsize = 1 we still include that dimension for compatability with batched linear bounds.
    """

    def __init__(self, Lambda: IntervalBounds, Omega: IntervalBounds, delta: IntervalBounds, theta: IntervalBounds):
        assert Lambda.dim() == 3, "Expected shape of [batchsize x m x n]"
        assert Omega.dim() == 3, "Expected shape of [batchsize x m x n]"
        assert delta.dim() == 2, "Expected shape of [batchsize x m]"
        assert theta.dim() == 2, "Expected shape of [batchsize x m]"
        assert Lambda.shape[0] == Omega.shape[0] == delta.shape[0] == theta.shape[0]
        assert Lambda.shape[1] == Omega.shape[1] == delta.shape[1] == theta.shape[1]
        assert Lambda.shape[2] == Omega.shape[2]

        self.Lambda = Lambda
        self.Omega = Omega
        self.delta = delta
        self.theta = theta

    def concretize(self, input_bounds: IntervalBounds) -> IntervalBounds:
        """
        Concretize the linear bounds using the input region.
        Args:
            input_region (IntervalBounds): Bounds on the input x.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the output.
        """
        assert input_bounds.dim() == 2, "Expected shape of [batchsize x n]"
        assert input_bounds.shape[1] == self.Lambda.shape[2]

        lower = ((self.Omega @ input_bounds.unsqueeze(-1)).squeeze(-1) + self.theta).lb
        upper = ((self.Lambda @ input_bounds.unsqueeze(-1)).squeeze(-1) + self.delta).ub
        return IntervalBounds(lower, upper)

    def __iter__(self) -> Iterator[IntervalBounds]:
        """Implement iter for unpacking."""
        return iter((self.Lambda, self.Omega, self.delta, self.theta))
