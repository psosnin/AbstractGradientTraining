from typing import Optional

import torch
from abstract_gradient_training.interval_tensor import IntervalTensor
from abstract_gradient_training import bound_utils


@torch.no_grad()
def bound_forward_pass(param_l, param_u, x0_l, x0_u, relu_lb="zero", **kwargs):
    """
    Compute intermediate bounds on the network using the abstract transformer implementation of the CROWN algorithm
    given the input domain.
    Parameters:
        param_l: list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u: list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l: [batchsize x input_dim x 1] lower bound on the input batch to the network
        x0_u: [batchsize x input_dim x 1] upper bound on the input batch to the network
        relu_lb: lower bound to use for the ReLU activation (one of "zero", "one", "parallel")
    Returns:
        logit_l: lower bounds on the logits
        logit_u: upper bounds on the logits
        inter_l: list of lower bounds on the intermediate activations (input interval, then post-relu bounds)
        inter_u: list of upper bounds on the intermediate activations (input interval, then post-relu bounds)
    """
    # validate the input
    param_l, param_u, x0_l, x0_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)

    # form IntervalTensors of the parameters
    W = [IntervalTensor(W_l, W_u) for W_l, W_u in zip(param_l[::2], param_u[::2])]
    b = [IntervalTensor(b_l, b_u) for b_l, b_u in zip(param_l[1::2], param_u[1::2])]

    # initialise the input node
    input_interval = IntervalTensor(x0_l, x0_u)
    x = CrownNode(None, None, None, None, None,  input_interval)
    bounds = [input_interval]

    # perform forward pass
    for (Wk, bk) in zip(W, b):
        xhat = AffineNode(x, Wk, bk)
        x = ReLUNode(xhat, relu_lb=relu_lb)
        bounds.append(xhat.concretize().relu())

    # format results to match the other implementations
    bounds.pop()  # remove the last bound as it shouldn't have a relu and instead is returned as logit_l, logit_u
    inter_l = [b.lb for b in bounds]  # get lower and upper bounds in lists
    inter_u = [b.ub for b in bounds]
    logit_l, logit_u = xhat.concretize().tuple

    return logit_l, logit_u, inter_l, inter_u


@torch.no_grad()
def bound_backward_pass(dL_min, dL_max, param_l, param_u, inter_l, inter_u, **kwargs):
    """
    Compute bounds on the gradient using the abstract transformer implementation of the CROWN algorithm.
    Parameters:
        dL_min: [batchsize x output_dim x 1] lower bound on the gradient of the loss with respect to the logits
        dL_max: [batchsize x output_dim x 1] upper bound on the gradient of the loss with respect to the logits
        param_l: list of lower bounds on the weights and biases given as a list [W1, b1, ..., Wm, bm]
        param_u: list of upper bounds on the weights and biases given as a list [W1, b1, ..., Wm, bm]
        inter_l: list of lower bounds on the intermediate activations given as a list [x0, ..., xL]
        inter_u: list of upper bounds on the intermediate activations given as a list [x0, ..., xL]
    Returns:
        grads_l: list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u: list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
    """
    # validate the input
    dL_min, dL_max, param_l, param_u, inter_l, inter_u = bound_utils.validate_backward_bound_input(
        dL_min, dL_max, param_l, param_u, inter_l, inter_u
    )

    # form IntervalTensors of the parameters and intermediate bounds
    W = [IntervalTensor(W_l, W_u) for W_l, W_u in zip(param_l[::2], param_u[::2])]
    inter = [IntervalTensor(l, u) for l, u in zip(inter_l, inter_u)]

    # define the first node in the crown computation graph
    dL_dxhat = CrownNode(None, None, None, None, None, IntervalTensor(dL_min, dL_max))
    grads_l, grads_u = [], []

    # compute all partials wrt the hidden layers
    for i in range(len(W) - 1, -1, -1):
        dL_dW = dL_dxhat.concretize() * inter[i].T()
        grads_l.extend([dL_dxhat.concretize().lb, dL_dW.lb])
        grads_u.extend([dL_dxhat.concretize().ub, dL_dW.ub])
        if i == 0:
            break
        dL_dx = AffineNode(dL_dxhat, W[i].T())
        dL_dxhat = MulNode(dL_dx, inter[i].heaviside())

    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u


class CrownNode:
    """
    A node representing an abstract domain consisting of linear bounds on a function out_var = f(in_var) such that
        out_var <= Lambda @ in_var + delta <= upper
        out_var >= Omega @ in_var + theta >= lower
    where the coefficients Lambda, Omega and biases delta, theta are all intervals.
    NOTE: The backpropagation procedure in this node can handle general linear bounds, but can be made much more
    efficient for specific cases, e.g. when Lambda = Omega or diagonal Lambda, Omega.
    """

    def __init__(self, in_var, Lambda, delta, Omega, theta, conc=None):
        """
        Parameters:
            in_var: CrownNode representing the input node
            Lambda: IntervalTensor representing bounds on the coefficient matrix Lambda
            delta: IntervalTensor representing bounds on the bias delta
            Omega: IntervalTensor representing bounds on the coefficient matrix Omega
            theta: IntervalTensor representing bounds on the bias theta
            conc: IntervalTensor representing concrete bounds on this node
        """
        assert isinstance(Lambda, Optional[IntervalTensor])
        assert isinstance(Omega, Optional[IntervalTensor])
        assert isinstance(delta, Optional[torch.Tensor])
        assert isinstance(theta, Optional[torch.Tensor])
        assert isinstance(conc, Optional[IntervalTensor])

        self.in_var = in_var
        self.Lambda = Lambda
        self.Omega = Omega
        self.delta = delta
        self.theta = theta
        self.conc = conc

        # check that we include the batch dimension
        if in_var is not None:
            assert len(self.Lambda.shape) == 3
            assert len(self.Omega.shape) == 3
            assert len(self.delta.shape) == 3
            assert len(self.theta.shape) == 3

    @torch.no_grad()
    def backpropagate(self):
        """
        Backpropagate the linear bounds from this node back to the input node x0 to get linear bounds of the form
            Omega0 @ x0 + theta0 <= out_var <= Lambda0 @ x0 + delta0
        The bound is then concretized using interval arithmetic.
        """
        Lambda0, delta0 = self.Lambda, self.delta
        Omega0, theta0 = self.Omega, self.theta

        cur = self.in_var
        # backpropagate the bounds from the current node to its input node, until we reach a node without an input
        while cur.in_var is not None:
            theta0 = cur._backpropTheta(Omega0, theta0)
            delta0 = cur._backpropDelta(Lambda0, delta0)
            Lambda0, delta0 = cur._backpropLambda(Lambda0, delta0)
            Omega0, theta0 = cur._backpropOmega(Omega0, theta0)
            cur = cur.in_var
        # concretize the bounds using the concrete bounds on cur, which should now be the input variable
        upper = (Lambda0 @ cur.conc + delta0).ub
        lower = (Omega0 @ cur.conc + theta0).lb
        self.conc = IntervalTensor(lower, upper)

    def concretize(self):
        """
        Return a tuple of concrete lower and upper bounds on the value of this node.
        """
        if self.conc is None:
            self.backpropagate()
        return self.conc

    def _backpropLambda(self, Lambda0, delta0):
        """
        Helper function for the backpropagation procedure that computes the new Lambda0 using
            Lambda0 = ((Lambda0 * Lambda) * (Lambda0 > 0) + (Lambda0 * Omega) * (Lambda0 < 0)).sum()
        """
        # compute sign masks
        pos_mask = (Lambda0.lb.unsqueeze(-1) > 0)  # lower bound greater than 0
        neg_mask = (Lambda0.ub.unsqueeze(-1) < 0)  # upper bound less than 0
        arb_mask = (Lambda0.lb <= 0) & (Lambda0.ub >= 0)  # sign arbitrary
        # handle case where Lambda0 spans 0
        delta0 = delta0 + ((Lambda0 * arb_mask) @ self.concretize()).ub
        # handle cases where the sign of Lambda0 is fixed
        Lambda0 = (
            ((Lambda0.unsqueeze(-1) * self.Lambda.unsqueeze(1)) * pos_mask).sum(-2) +
            ((Lambda0.unsqueeze(-1) * self.Omega.unsqueeze(1)) * neg_mask).sum(-2)
        )
        return Lambda0, delta0

    def _backpropOmega(self, Omega0, theta0):
        """
        Helper function for the backpropagation procedure that computes the new Omega0 using
            Omega0 = ((Omega0 * Omega) * (Omega0 > 0) + (Omega0 * Lambda) * (Omega0 < 0)).sum()
        """
        pos_mask = (Omega0.lb.unsqueeze(-1) > 0)
        neg_mask = (Omega0.ub.unsqueeze(-1) < 0)
        arb_mask = (Omega0.lb <= 0) & (Omega0.ub >= 0)
        # handle case where Omega0 spans 0
        theta0 = theta0 + ((Omega0 * arb_mask) @ self.concretize()).lb
        # handle cases where the sign of Omega0 is fixed
        Omega0 = (
            ((Omega0.unsqueeze(-1) * self.Omega.unsqueeze(1)) * pos_mask).sum(-2) +
            ((Omega0.unsqueeze(-1) * self.Lambda.unsqueeze(1)) * neg_mask).sum(-2)
        )
        return Omega0, theta0

    def _backpropDelta(self, Lambda0, delta0):
        """
        Helper function for the backpropagation procedure that computes the new delta0 using
            delta0 = delta0 + ((Lambda0 * delta) * (Lambda0 > 0) + (Lambda0 * theta) * (Lambda0 < 0)).sum()
        """
        # the case where the sign of Lambda0 is arbitrary is already accounted for in the _backpropLambda function
        delta0 = delta0 + (
            (Lambda0.unsqueeze(-1) * self.delta.unsqueeze(1) * (Lambda0.lb.unsqueeze(-1) > 0)).sum(-2).ub +
            (Lambda0.unsqueeze(-1) * self.theta.unsqueeze(1) * (Lambda0.ub.unsqueeze(-1) < 0)).sum(-2).ub
        )
        return delta0

    def _backpropTheta(self, Omega0, theta0):
        """
        Helper function for the backpropagation procedure that computes the new theta0 using
            theta0 = theta0 + ((Omega0 * theta) * (Omega0 > 0) + (Omega0 * delta) * (Omega0 < 0)).sum()
        """
        # the case where the sign of Omega0 is arbitrary is already accounted for in the _backpropOmega function
        theta0 = theta0 + (
            (Omega0.unsqueeze(-1) * self.theta.unsqueeze(1) * (Omega0.lb.unsqueeze(-1) > 0)).sum(-2).lb +
            (Omega0.unsqueeze(-1) * self.delta.unsqueeze(1) * (Omega0.ub.unsqueeze(-1) < 0)).sum(-2).lb
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

    def __init__(self, in_var, W, b=None):
        """
        Parameters:
        in_var: CrownNode representing the input node
        W: tuple (W_l, W_u) representing bounds on the weight matrix
        b: tuple (b_l, b_u) representing bounds on the delta
        """
        device = W[0].get_device()
        device = torch.device(device) if device != -1 else torch.device('cpu')
        if b is None:
            b = IntervalTensor.zeros(W.shape[-2], 1, dtype=W.dtype, device=device)
        # add batch dimension to the node
        self.W = W.unsqueeze(0) if len(W.shape) == 2 else W
        self.b = b.unsqueeze(0) if len(b.shape) == 2 else b
        self.in_var = in_var
        super().__init__(in_var, self.W, self.b.ub, self.W, self.b.lb)

    def _backpropLambda(self, Lambda0, delta0):
        """
        Helper function for the backpropagation procedure that computes the new Lambda0
        """
        arb_mask = (Lambda0.lb <= 0) & (Lambda0.ub >= 0)
        # handle case where Lambda0 spans 0
        delta0 = delta0 + ((Lambda0 * arb_mask) @ self.concretize()).ub
        # handle cases where the sign of Lambda0 is fixed
        Lambda0 = (Lambda0 * (~arb_mask)) @ self.W
        return Lambda0, delta0

    def _backpropOmega(self, Omega0, theta0):
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

    def __init__(self, in_var, relu_lb="zero"):
        """
        Parameters:
            in_var: CrownNode input to the ReLU node
        """
        x = in_var.concretize()
        device = x.get_device()
        device = torch.device(device) if device != -1 else torch.device('cpu')

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
        beta_u[I] = - x.lb[I]

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

    def _backpropLambda(self, Lambda0, delta0):
        """
        Helper function for the backpropagation procedure that computes the new Lambda0
        """
        delta0 += ((Lambda0 * ((Lambda0.ub >= 0) & (Lambda0.lb <= 0))) @ self.concretize()).ub
        Lambda0 = (
            (Lambda0 * self.alpha_u.transpose(-2, -1)) * (Lambda0.lb > 0) +
            (Lambda0 * self.alpha_l.transpose(-2, -1)) * (Lambda0.ub < 0)
        )
        return Lambda0, delta0

    def _backpropOmega(self, Omega0, theta0):
        """
        Helper function for the backpropagation procedure that computes the new Omega0
        """
        theta0 += ((Omega0 * ((Omega0.ub >= 0) & (Omega0.lb <= 0))) @ self.concretize()).lb
        Omega0 = (
            (Omega0 * self.alpha_l.transpose(-2, -1)) * (Omega0.lb > 0) +
            (Omega0 * self.alpha_u.transpose(-2, -1)) * (Omega0.ub < 0)
        )
        return Omega0, theta0


class MulNode(CrownNode):
    """
    A node in the CROWN computation graph consisting of the elementwise multiplication out_var = s * in_var for s in
    [l, u]. Unlike the more general CROWN nodes, the lower and upper linear bounds are the same here but still diagonal,
    so we can use tricks from the ReLUNode and the AffineNode.
    """

    def __init__(self, in_var, s):
        """
        Parameters:
        in_var: CrownNode representing the input node
        """
        # Validate input
        assert isinstance(s, IntervalTensor)
        self.s = s
        delta = torch.zeros_like(s.lb)
        # define the diagonal matrix Lambda from s
        Lambda = IntervalTensor(
            torch.diag_embed(self.s.lb.flatten(start_dim=1), dim1=-2, dim2=-1),
            torch.diag_embed(self.s.ub.flatten(start_dim=1), dim1=-2, dim2=-1)
        )
        super().__init__(in_var, Lambda, delta, Lambda, delta)

    def _backpropLambda(self, Lambda0, delta0):
        """
        Helper function for the backpropagation procedure that computes the new Lambda0
        """
        return Lambda0 * self.s.transpose(-2, -1), delta0

    def _backpropOmega(self, Omega0, theta0):
        """
        Helper function for the backpropagation procedure that computes the new Omega0
        """
        return Omega0 * self.s.transpose(-2, -1), theta0

    def _backpropDelta(self, Lambda0, delta0):
        """
        Helper function for the backpropagation procedure that computes the new delta0. The bias for this node is zero
        so we simply pass the current delta0 forward.
        """
        return delta0

    def _backpropTheta(self, Omega0, theta0):
        """
        Helper function for the backpropagation procedure that computes the new theta0. The bias for this node is zero
        so we simply pass the current theta0 forward.
        """
        return theta0
