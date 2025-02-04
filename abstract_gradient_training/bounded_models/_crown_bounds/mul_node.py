""" Class representing nodes in the linear bound propagation computational graph. """

from __future__ import annotations

from typing import Literal

import torch

from abstract_gradient_training.bounded_models._crown_bounds import Node, LinearBounds, IntervalBounds


class MulNode(Node):
    """
    A node in the CROWN computation graph consisting of the elementwise multiplication out_var = s * in_var for s in
    [s_l, s_u]. Unlike the more general CROWN nodes, the lower and upper linear bounds are the same here but still
    diagonal, so we can use tricks from the ReLUNode and the AffineNode.
    """

    def __init__(
        self,
        in_var: Node,
        s_l: torch.Tensor,
        s_u: torch.Tensor,
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
    ):
        """
        Args:
            in_var (Node): representing the input node.
            s_l (torch.Tensor): Lower bound on the coefficients.
            s_u (torch.Tensor): Upper bound on the coefficients.
        """
        super().__init__()
        # Validate input
        if not isinstance(s_l, torch.Tensor) or not isinstance(s_u, torch.Tensor):
            raise ValueError("s_l and s_u must be torch.Tensor")
        if s_l.dim() != 2 or s_u.dim() != 2:
            raise ValueError("s_l and s_u must be 2D tensors")
        self.s = IntervalBounds(s_l, s_u, interval_matmul=interval_matmul)
        self.in_var = in_var
        self.interval_matmul: Literal["rump", "exact", "nguyen"] = interval_matmul

    def _backpropagate(self, backward_bounds: LinearBounds) -> None:
        """
        Extend the linear bounds passed in as an argument to include this node. Say the computational graph represented
        has the following structure:
            x0 -> x1 -> ... -> xn-1 -> xn -> xn+1 -> ... -> xN
        where
            - the current node (self) represents bounds L @ xn + d <= xn+1 <= O @ xn + t
            - the argument bounds (backward bounds) represents bounds L @ xn-1 + d <= xN <= O @ xn-1 + t
        then return linear bounds from xn-1 to xN of the form
            L @ xn-1 + d <= xN <= O @ xn-1 + t
        """
        backward_bounds.Lambda *= self.s.unsqueeze(1)
        backward_bounds.Omega *= self.s.unsqueeze(1)

    def _init_backpropagation(self) -> LinearBounds:
        """
        Initialise the coefficients for performing backpropagation from this node.
        """
        Lambda = Omega = IntervalBounds(
            torch.diag_embed(self.s.lb.flatten(start_dim=1), dim1=-2, dim2=-1),
            torch.diag_embed(self.s.ub.flatten(start_dim=1), dim1=-2, dim2=-1),
            interval_matmul=self.interval_matmul,
        )
        delta = IntervalBounds.zeros_like(self.s)
        theta = IntervalBounds.zeros_like(self.s)
        return LinearBounds(Lambda, Omega, delta, theta)


if __name__ == "__main__":
    from abstract_gradient_training.bounded_models._crown_bounds.base_node import InputNode

    x = torch.randn(5, 3)
    s1 = torch.randn(5, 3)
    s2 = torch.randn(5, 3)
    print(x * s1 * s2)

    x = MulNode(InputNode(x, x), s1, s1)
    x = MulNode(x, s2, s2)
    l, u = x.concretize()
    assert torch.allclose(l, u)
    print(l)
