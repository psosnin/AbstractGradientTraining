""" Class representing nodes in the linear bound propagation computational graph. """

from __future__ import annotations

from typing import Literal

import torch

from abstract_gradient_training.bounded_models._crown_bounds import Node, LinearBounds, IntervalBounds


class ReLUNode(Node):
    """
    A node in the CROWN computation graph representing out_var = ReLU(in_var). The bounds on the output are
            out_var <= Lambda @ in_var + delta <= upper
            out_var >= Omega @ in_var + theta >= lower
    where unlike the general CROWN node, Lambda and Omega are diagonal and constant (i.e. not interval matrices).
    This allows for more efficient backpropagation.
    The linear relaxation of the ReLU activation function is given by

            alpha_l * x + beta_l <= ReLU(x) <= alpha_u * x + beta_u

    This code is partially inspired by https://github.com/Zinoex/bound_propagation
    """

    def __init__(
        self,
        in_var: Node,
        relu_lb: Literal["one", "zero", "parallel", "optimizable"] = "zero",
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
    ):
        """
        Args:
            in_var (Node): Input to the ReLU node
            relu_lb (str): Lower bound to use for the ReLU activation (one of "zero", "one", "parallel")
            interval_matmul (str): Method to use for interval matrix multiplication
        """
        super().__init__()
        assert relu_lb in ["zero", "one", "parallel", "optimizable"], f"Unknown relu lower bound function: {relu_lb}"
        assert interval_matmul in ["rump", "exact", "nguyen"], f"Unknown interval matmul method: {interval_matmul}"
        self.in_var = in_var
        self.interval_matmul: Literal["rump", "exact", "nguyen"] = interval_matmul
        self.relu_lb = relu_lb
        self.initialized = False
        self.update_relaxation()

    def update_relaxation(self) -> None:
        """
        Set the alpha-beta relaxation coefficients for this node, i.e.
            alpha_l * x + beta_l <= ReLU(x) <= alpha_u * x + beta_u
        Note that beta_l is not set since it is always 0.
        """
        x_l, x_u = self.in_var.concretize()  # type: ignore
        assert x_l.shape == x_u.shape
        assert x_l.dim() == 2, "Expected input to be a 2D tensor"
        # relu activation sets
        neg = x_u <= 0
        pos = x_l >= 0
        pos_neg = (x_l < 0) & (x_u > 0)
        zero_width = torch.isclose(x_l, x_u, rtol=0.0, atol=1e-8)

        # initialize the relaxation coefficients
        self.alpha_l = torch.zeros_like(x_l)
        self.alpha_u = torch.zeros_like(x_l)
        self.beta_u = torch.zeros_like(x_l)

        # handle zero width regime, 0.0 * x + ReLU(x) <= ReLU(x) <= 0.0 * x + ReLU(x)
        self.alpha_l[zero_width] = 0.0
        self.alpha_u[zero_width] = 0.0
        self.beta_u[zero_width] = torch.relu(x_u[zero_width])

        # handle negative regime, 0.0 * x + 0.0 <= ReLU(x) = 0 <= 0.0 * x + 0.0
        self.alpha_l[neg] = 0.0
        self.alpha_u[neg] = 0.0
        self.beta_u[neg] = 0.0

        # handle positive regime, 1.0 * x + 0.0 <= ReLU(x) = x <= 1.0 * x + 0.0
        self.alpha_l[pos] = 1.0
        self.alpha_u[pos] = 1.0
        self.beta_u[pos] = 0.0

        # handle mixed regime, alpha_l * x + beta_l <= ReLU(x) <= alpha_u * x + beta_u
        self.alpha_u[pos_neg] = x_u[pos_neg] / (x_u[pos_neg] - x_l[pos_neg])
        self.beta_u[pos_neg] = -x_l[pos_neg] * self.alpha_u[pos_neg]

        if not self.initialized and self.relu_lb == "optimizable":  # only do this the first time
            self.alpha_l_opt = self.alpha_l.clone().detach().requires_grad_()
            self._optimizable_params.append(self.alpha_l_opt)

        if self.relu_lb == "one":
            self.alpha_l[pos_neg] = 1.0
        elif self.relu_lb == "parallel":
            self.alpha_l = self.alpha_u
        elif self.relu_lb == "optimizable":
            self.alpha_l_opt_mask = pos_neg
            self.alpha_l[self.alpha_l_opt_mask] = self.alpha_l_opt[self.alpha_l_opt_mask].clamp(min=0.0, max=1.0)
        else:
            assert self.relu_lb == "zero"

        # check that the slopes are valid
        assert (self.alpha_l >= 0.0).all() and (self.alpha_l <= 1.0).all()
        assert (self.alpha_u >= 0.0).all() and (self.alpha_u <= 1.0).all()
        self.initialized = True

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
        self.update_relaxation()
        Lambda, Omega, delta, theta = backward_bounds
        conc = self.concretize()

        # backpropagate the upper bound
        pos_mask = Lambda.lb >= 0
        neg_mask = Lambda.ub <= 0
        mixed_mask = (Lambda.lb < 0) & (Lambda.ub > 0)
        backward_bounds.delta = (
            delta
            + (Lambda * self.beta_u.unsqueeze(1) * pos_mask).sum(-1)
            # + (Lambda * self.beta_l.unsqueeze(1) * neg_mask).ub.sum(-1)  # beta_l = 0
            + ((Lambda * mixed_mask) @ conc.unsqueeze(-1)).squeeze(-1)
        )
        backward_bounds.Lambda = Lambda * (self.alpha_u.unsqueeze(1) * pos_mask + self.alpha_l.unsqueeze(1) * neg_mask)

        # backpropagate the lower bound
        pos_mask = Omega.lb >= 0
        neg_mask = Omega.ub <= 0
        mixed_mask = (Omega.lb < 0) & (Omega.ub > 0)
        backward_bounds.theta = (
            theta
            # + (Omega0 * self.beta_l.unsqueeze(1) * pos_mask).lb.sum(-1)  # beta_l = 0
            + (Omega * self.beta_u.unsqueeze(1) * neg_mask).sum(-1)
            + ((Omega * mixed_mask) @ conc.unsqueeze(-1)).squeeze(-1)
        )
        backward_bounds.Omega = Omega * (self.alpha_l.unsqueeze(1) * pos_mask + self.alpha_u.unsqueeze(1) * neg_mask)

    def _init_backpropagation(self) -> LinearBounds:
        """
        Initialise the coefficients for performing backpropagation from this node.
        """
        self.update_relaxation()
        Lambda0 = IntervalBounds(
            torch.diag_embed(self.alpha_u.flatten(start_dim=1), dim1=-2, dim2=-1), interval_matmul=self.interval_matmul
        )
        Omega0 = IntervalBounds(
            torch.diag_embed(self.alpha_l.flatten(start_dim=1), dim1=-2, dim2=-1), interval_matmul=self.interval_matmul
        )
        delta0 = IntervalBounds((self.alpha_u * self.beta_u).clone())
        theta0 = IntervalBounds.zeros_like(delta0)
        return LinearBounds(Lambda0, Omega0, delta0, theta0)


if __name__ == "__main__":
    from abstract_gradient_training.bounded_models._crown_bounds import InputNode, AffineNode

    batchsize = 20
    shapes = [5, 4, 3, 2]

    x_conc = torch.randn(batchsize, shapes[0], device="cuda:1")
    x = InputNode(x_conc - 0.2, x_conc + 0.2)

    for i in range(1, len(shapes)):
        W = torch.randn(shapes[i], shapes[i - 1], device="cuda:1")
        b = torch.randn(shapes[i], device="cuda:1")
        x = AffineNode(x, W - 0.01, W + 0.01, b - 0.01, b + 0.01)
        x = ReLUNode(x, relu_lb="optimizable")
        x_conc = torch.relu(x_conc @ W.T + b)

    l, u = x.concretize()
    assert (l <= u).all()
    assert (l <= x_conc).all()
    assert (u >= x_conc).all()

    optimizer = torch.optim.Adam(x.optimizable_parameters(), lr=1e-1)  # type: ignore
    for i in range(50):
        optimizer.zero_grad()
        x.clear_cached()
        l, u = x.concretize()
        assert (l <= u).all()
        assert (l <= x_conc).all()
        assert (u >= x_conc).all()
        loss = (u - l).sum()
        print(i, loss)
        loss.backward()
        optimizer.step()
