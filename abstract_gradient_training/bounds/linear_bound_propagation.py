"""Linear bound propagation (CROWN)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from abstract_gradient_training.bounds.interval_tensor import IntervalTensor
from abstract_gradient_training.bounds import bound_utils


@torch.no_grad()
def bound_forward_pass(
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    x0_l: torch.Tensor,
    x0_u: torch.Tensor,
    relu_lb: str = "zero",
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using the double-interval crown algorithm

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.
        relu_lb (str): lower bound to use for the ReLU activation (one of "zero", "one", "parallel")

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
    """

    # validate the input
    param_l, param_u, x0_l, x0_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)

    # form IntervalTensors of the parameters
    W = [IntervalTensor(W_l, W_u) for W_l, W_u in zip(param_l[::2], param_u[::2])]
    b = [IntervalTensor(b_l, b_u) for b_l, b_u in zip(param_l[1::2], param_u[1::2])]

    # initialise the input node
    input_interval = IntervalTensor(x0_l, x0_u)
    x = CrownNode(None, None, None, None, None, input_interval)
    bounds = [input_interval]

    # perform forward pass
    for Wk, bk in zip(W, b):
        xhat = AffineNode(x, Wk, bk)
        x = ReLUNode(xhat, relu_lb=relu_lb)
        bounds.append(xhat.concretize())

    # format results to match the other implementations
    activations_l = [b.lb for b in bounds]
    activations_u = [b.ub for b in bounds]

    return activations_l, activations_u


def bound_backward_pass(
    dL_min: torch.Tensor,
    dL_max: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    activations_l: list[torch.Tensor],
    activations_u: list[torch.Tensor],
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters, intermediate activations and the first partial derivative of the loss, compute
    bounds on the gradients of the loss with respect to the parameters of the network using double-interval crown
    algorithm.

    Args:
        dL_min (torch.Tensor): lower bound on the gradient of the loss with respect to the logits
        dL_max (torch.Tensor): upper bound on the gradient of the loss with respect to the logits
        param_l (list[torch.Tensor]): list of lower bounds on the parameters [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters [W1, b1, ..., Wm, bm]
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].

    Returns:
        grads_l (list[torch.Tensor]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[torch.Tensor]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
    """
    # validate the input
    dL_min, dL_max, param_l, param_u, activations_l, activations_u = bound_utils.validate_backward_bound_input(
        dL_min, dL_max, param_l, param_u, activations_l, activations_u
    )

    # convert pre-relu activations to post-relu activations
    activations_l = [activations_l[0]] + [F.relu(x) for x in activations_l[1:-1]] + [activations_l[-1]]
    activations_u = [activations_u[0]] + [F.relu(x) for x in activations_u[1:-1]] + [activations_u[-1]]

    # form IntervalTensors of the parameters and intermediate bounds
    W = [IntervalTensor(W_l, W_u) for W_l, W_u in zip(param_l[::2], param_u[::2])]
    activations = [IntervalTensor(l, u) for l, u in zip(activations_l, activations_u)]

    # define the first node in the crown computation graph
    dL_dxhat = CrownNode(None, None, None, None, None, IntervalTensor(dL_min, dL_max))
    grads_l, grads_u = [], []

    # compute all partials wrt the hidden layers
    for i in range(len(W) - 1, -1, -1):
        dL_dW = dL_dxhat.concretize() * activations[i].T()
        grads_l.extend([dL_dxhat.concretize().lb, dL_dW.lb])
        grads_u.extend([dL_dxhat.concretize().ub, dL_dW.ub])
        if i == 0:
            break
        dL_dx = AffineNode(dL_dxhat, W[i].T())
        dL_dxhat = MulNode(dL_dx, activations[i].heaviside())

    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u


class CrownNode:
    """
    A node representing an abstract domain consisting of linear bounds on a function out_var = f(in_var) such that
        out_var <= Lambda @ in_var + delta <= upper
        out_var >= Omega @ in_var + theta >= lower
    where the coefficients Lambda, Omega and biases delta, theta are all intervals.

    Example:
        Given a sequence of nodes x0 => x1 => x2, the following linear equalities hold for all x0 in x0.conc:
            x1.Lambda @ x0 + x1.delta <= x1 <= x1.Lambda @ x0 + x1.delta
            x2.Lambda @ x1 + x2.delta <= x2 <= x2.Lambda @ x1 + x2.delta
        The backpropagation procedure then uses linear bound propagation to compute a set of linear inequalities from
        x2 to x0:
            Omega0 @ x0 + theta0 <= x0 <= Lambda0 @ x0 + delta0
        where Omega0, theta0, Lambda0, delta0 are computed from the backpropagation procedure.

    NOTE: The backpropagation procedure in this node can handle general linear bounds, but can be made much more
    efficient for specific cases, e.g. when Lambda = Omega or diagonal for Lambda, Omega.
    """

    def __init__(
        self,
        in_var: CrownNode | None,
        Lambda: IntervalTensor | None,
        delta: torch.Tensor | None,
        Omega: IntervalTensor | None,
        theta: torch.Tensor | None,
        conc: IntervalTensor | None = None,
    ):
        """
        Initialise the CrownNode with the given linear bounds relating the output to input of the node.
        Args:
            in_var (CrownNode | None): CrownNode representing the input node
            Lambda (IntervalTensor | None): Bounds on the coefficient matrix Lambda
            delta (IntervalTensor | None): Bounds on the bias delta
            Omega (IntervalTensor | None): Bounds on the coefficient matrix Omega
            theta (IntervalTensor | None): Bounds on the bias theta
            conc (IntervalTensor | None): Bounds on the concretization of this node
        """
        # TODO: change the 'input' node to simply be an IntervalTensor assigned to in_var, so we don't need to have
        # all these IntervalTensor | None types
        assert isinstance(Lambda, IntervalTensor | None)
        assert isinstance(Omega, IntervalTensor | None)
        assert isinstance(delta, torch.Tensor | None)
        assert isinstance(theta, torch.Tensor | None)
        assert isinstance(conc, IntervalTensor | None)

        self.in_var = in_var
        self.Lambda = Lambda
        self.Omega = Omega
        self.delta = delta
        self.theta = theta
        self.conc = conc

        # check that we include the batch dimension
        if in_var is not None:
            assert self.Lambda is not None and len(self.Lambda.shape) == 3
            assert self.Omega is not None and len(self.Omega.shape) == 3
            assert self.delta is not None and len(self.delta.shape) == 3
            assert self.theta is not None and len(self.theta.shape) == 3

    @torch.no_grad()
    def backpropagate(self) -> None:
        """
        Backpropagate the linear bounds from this node back to the input node x0 to get linear bounds of the form
            Omega0 @ x0 + theta0 <= out_var <= Lambda0 @ x0 + delta0
        The bound is then concretized using interval arithmetic.
        """
        Lambda0, delta0 = self.Lambda, self.delta
        Omega0, theta0 = self.Omega, self.theta

        cur = self.in_var
        if cur is None:
            return  # we are already at the input node

        # backpropagate the bounds from the current node to its input node, until we reach a node without an input
        while cur.in_var is not None:
            theta0 = cur.backprop_Theta(Omega0, theta0)
            delta0 = cur.backprop_Delta(Lambda0, delta0)
            Lambda0, delta0 = cur.backprop_Lambda(Lambda0, delta0)
            Omega0, theta0 = cur.backprop_Omega(Omega0, theta0)
            cur = cur.in_var
        # concretize the bounds using the concrete bounds on cur, which should now be the input variable
        upper = (Lambda0 @ cur.conc + delta0).ub
        lower = (Omega0 @ cur.conc + theta0).lb
        self.conc = IntervalTensor(lower, upper)

    def concretize(self) -> IntervalTensor:
        """
        Return a tuple of concrete lower and upper bounds on the value of this node.
        """
        if self.conc is None:
            self.backpropagate()
        return self.conc

    def backprop_Lambda(self, Lambda0: IntervalTensor, delta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Lambda0 using
            Lambda0 = ((Lambda0 * Lambda) * (Lambda0 > 0) + (Lambda0 * Omega) * (Lambda0 < 0)).sum()
        """
        # compute sign masks
        pos_mask = Lambda0.lb.unsqueeze(-1) > 0  # lower bound greater than 0
        neg_mask = Lambda0.ub.unsqueeze(-1) < 0  # upper bound less than 0
        arb_mask = (Lambda0.lb <= 0) & (Lambda0.ub >= 0)  # sign arbitrary
        # handle case where Lambda0 spans 0
        delta0 = delta0 + ((Lambda0 * arb_mask) @ self.concretize()).ub
        # handle cases where the sign of Lambda0 is fixed
        Lambda0 = ((Lambda0.unsqueeze(-1) * self.Lambda.unsqueeze(1)) * pos_mask).sum(-2) + (
            (Lambda0.unsqueeze(-1) * self.Omega.unsqueeze(1)) * neg_mask
        ).sum(-2)
        return Lambda0, delta0

    def backprop_Omega(self, Omega0: IntervalTensor, theta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Omega0 using
            Omega0 = ((Omega0 * Omega) * (Omega0 > 0) + (Omega0 * Lambda) * (Omega0 < 0)).sum()
        """
        pos_mask = Omega0.lb.unsqueeze(-1) > 0
        neg_mask = Omega0.ub.unsqueeze(-1) < 0
        arb_mask = (Omega0.lb <= 0) & (Omega0.ub >= 0)
        # handle case where Omega0 spans 0
        theta0 = theta0 + ((Omega0 * arb_mask) @ self.concretize()).lb
        # handle cases where the sign of Omega0 is fixed
        Omega0 = ((Omega0.unsqueeze(-1) * self.Omega.unsqueeze(1)) * pos_mask).sum(-2) + (
            (Omega0.unsqueeze(-1) * self.Lambda.unsqueeze(1)) * neg_mask
        ).sum(-2)
        return Omega0, theta0

    def backprop_Delta(self, Lambda0: IntervalTensor, delta0: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the backpropagation procedure that computes the new delta0 using
            delta0 = delta0 + ((Lambda0 * delta) * (Lambda0 > 0) + (Lambda0 * theta) * (Lambda0 < 0)).sum()
        """
        # the case where the sign of Lambda0 is arbitrary is already accounted for in the backprop_Lambda function
        delta0 = delta0 + (
            (Lambda0.unsqueeze(-1) * self.delta.unsqueeze(1) * (Lambda0.lb.unsqueeze(-1) > 0)).sum(-2).ub
            + (Lambda0.unsqueeze(-1) * self.theta.unsqueeze(1) * (Lambda0.ub.unsqueeze(-1) < 0)).sum(-2).ub
        )
        return delta0

    def backprop_Theta(self, Omega0: IntervalTensor, theta0: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the backpropagation procedure that computes the new theta0 using
            theta0 = theta0 + ((Omega0 * theta) * (Omega0 > 0) + (Omega0 * delta) * (Omega0 < 0)).sum()
        """
        # the case where the sign of Omega0 is arbitrary is already accounted for in the backprop_Omega function
        theta0 = theta0 + (
            (Omega0.unsqueeze(-1) * self.theta.unsqueeze(1) * (Omega0.lb.unsqueeze(-1) > 0)).sum(-2).lb
            + (Omega0.unsqueeze(-1) * self.delta.unsqueeze(1) * (Omega0.ub.unsqueeze(-1) < 0)).sum(-2).lb
        )
        return theta0


class AffineNode(CrownNode):
    """
    A node in the CROWN computation graph consisting of an affine transformation out_var = W @ in_var + b
    where W and b are intervals. Unlike the more general CROWN nodes, the lower and upper linear bounds are
    the same here, allowing for a more efficient backpropagation:
            out_var <= Lambda @ in_var + delta = W @ in_var + b <= upper
            out_var >= Omega @ in_var + theta = W @ in_var + b >= lower
    """

    def __init__(self, in_var: CrownNode, W: IntervalTensor, b: IntervalTensor | None = None):
        """
        Args:
            in_var (CrownNode): CrownNode representing the input node
            W (IntervalTensor): Bounds on the weight matrix
            b (IntervalTensor | None): Bounds on the bias
        """
        device = W[0].get_device()
        device = torch.device(device) if device != -1 else torch.device("cpu")
        if b is None:
            b = IntervalTensor.zeros(W.shape[-2], 1, dtype=W.dtype, device=device)
        # add batch dimension to the node
        self.W = W.unsqueeze(0) if len(W.shape) == 2 else W
        self.b = b.unsqueeze(0) if len(b.shape) == 2 else b
        self.in_var = in_var
        super().__init__(in_var, self.W, self.b.ub, self.W, self.b.lb)

    def backprop_Lambda(self, Lambda0: IntervalTensor, delta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Lambda0
        """
        arb_mask = (Lambda0.lb <= 0) & (Lambda0.ub >= 0)
        # handle case where Lambda0 spans 0
        delta0 = delta0 + ((Lambda0 * arb_mask) @ self.concretize()).ub
        # handle cases where the sign of Lambda0 is fixed
        Lambda0 = (Lambda0 * (~arb_mask)) @ self.W
        return Lambda0, delta0

    def backprop_Omega(self, Omega0: IntervalTensor, theta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Omega0
        """
        arb_mask = (Omega0.lb <= 0) & (Omega0.ub >= 0)
        # handle case where Lambda0 spans 0
        theta0 = theta0 + ((Omega0 * arb_mask) @ self.concretize()).lb
        # handle cases where the sign of Lambda0 is fixed
        Omega0 = (Omega0 * (~arb_mask)) @ self.W
        return Omega0, theta0


class ReLUNode(CrownNode):
    """
    A node in the CROWN computation graph representing out_var = ReLU(in_var). The bounds on the output are
            out_var <= Lambda @ in_var + delta <= upper
            out_var >= Omega @ in_var + theta >= lower
    where unlike the general CROWN node, Lambda and Omega are diagonal and constant (i.e. not interval matrices).
    This allows for more efficient backpropagation.
    """

    def __init__(self, in_var: CrownNode, relu_lb: str = "zero"):
        """
        Args:
            in_var (CrownNode): Input to the ReLU node
            relu_lb (str): Lower bound to use for the ReLU activation (one of "zero", "one", "parallel")
        """
        x = in_var.concretize()
        device = x.get_device()
        device = torch.device(device) if device != -1 else torch.device("cpu")

        # relu activation sets
        I_pos = torch.where(x.lb > 0)
        I = torch.where((x.lb < 0) & (x.ub > 0))

        # coefficients for linear relaxation of relu: alpha_l @ (x + beta_l) <= relu(x) <= alpha_u @ (x + beta_u)
        self.alpha_u = torch.zeros_like(x.lb, device=device)
        self.alpha_l = torch.zeros_like(x.lb, device=device)
        beta_u = torch.zeros_like(x.lb, device=device)
        beta_l = torch.zeros_like(x.lb, device=device)

        # compute coefficients of the linear relaxations of each neuron in the previous layer
        self.alpha_u[I_pos] = 1
        self.alpha_l[I_pos] = 1
        self.alpha_u[I] = (x.ub / (x.ub - x.lb))[I]
        beta_u[I] = -x.lb[I]

        if relu_lb == "one":
            self.alpha_l[I] = 1
        elif relu_lb == "parallel":
            self.alpha_l = self.alpha_u
        else:
            assert relu_lb == "zero"

        # check that the slopes are valid
        assert (self.alpha_l >= 0).all()
        assert (self.alpha_u >= 0).all()

        # convert coefficient to interval tensors and pass to CrownNode initializer
        delta = self.alpha_u * beta_u
        theta = self.alpha_l * beta_l
        Lambda = IntervalTensor(torch.diag_embed(self.alpha_u.flatten(start_dim=1), dim1=-2, dim2=-1))
        Omega = IntervalTensor(torch.diag_embed(self.alpha_l.flatten(start_dim=1), dim1=-2, dim2=-1))

        super().__init__(in_var, Lambda, delta, Omega, theta)

    def backprop_Lambda(self, Lambda0: IntervalTensor, delta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Lambda0
        """
        delta0 += ((Lambda0 * ((Lambda0.ub >= 0) & (Lambda0.lb <= 0))) @ self.concretize()).ub
        Lambda0 = (Lambda0 * self.alpha_u.transpose(-2, -1)) * (Lambda0.lb > 0) + (
            Lambda0 * self.alpha_l.transpose(-2, -1)
        ) * (Lambda0.ub < 0)
        return Lambda0, delta0

    def backprop_Omega(self, Omega0: IntervalTensor, theta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Omega0
        """
        theta0 += ((Omega0 * ((Omega0.ub >= 0) & (Omega0.lb <= 0))) @ self.concretize()).lb
        Omega0 = (Omega0 * self.alpha_l.transpose(-2, -1)) * (Omega0.lb > 0) + (
            Omega0 * self.alpha_u.transpose(-2, -1)
        ) * (Omega0.ub < 0)
        return Omega0, theta0


class MulNode(CrownNode):
    """
    A node in the CROWN computation graph consisting of the elementwise multiplication out_var = s * in_var for s in
    [l, u]. Unlike the more general CROWN nodes, the lower and upper linear bounds are the same here but still diagonal,
    so we can use tricks from the ReLUNode and the AffineNode.
    """

    def __init__(self, in_var: CrownNode, s: IntervalTensor):
        """
        Args:
            in_var (CrownNode): representing the input node
            s (IntervalTensor): Bounds on the elementwise multiplication
        """
        # Validate input
        assert isinstance(s, IntervalTensor)
        self.s = s
        delta = torch.zeros_like(s.lb)
        # define the diagonal matrix Lambda from s
        Lambda = IntervalTensor(
            torch.diag_embed(self.s.lb.flatten(start_dim=1), dim1=-2, dim2=-1),
            torch.diag_embed(self.s.ub.flatten(start_dim=1), dim1=-2, dim2=-1),
        )
        super().__init__(in_var, Lambda, delta, Lambda, delta)

    def backprop_Lambda(self, Lambda0: IntervalTensor, delta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Lambda0
        """
        return Lambda0 * self.s.transpose(-2, -1), delta0

    def backprop_Omega(self, Omega0: IntervalTensor, theta0: torch.Tensor) -> tuple[IntervalTensor, torch.Tensor]:
        """
        Helper function for the backpropagation procedure that computes the new Omega0
        """
        return Omega0 * self.s.transpose(-2, -1), theta0

    def backprop_Delta(self, Lambda0: IntervalTensor, delta0: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the backpropagation procedure that computes the new delta0. The bias for this node is zero
        so we simply pass the current delta0 forward.
        """
        return delta0

    def backprop_Theta(self, Omega0: IntervalTensor, theta0: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the backpropagation procedure that computes the new theta0. The bias for this node is zero
        so we simply pass the current theta0 forward.
        """
        return theta0
