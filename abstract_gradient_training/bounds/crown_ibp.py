"""Bounds that combine the double interval bound propagation and the double-interval crown algorithms."""

import torch

from abstract_gradient_training.bounds import interval_bound_propagation as ibp
from abstract_gradient_training.bounds import crown


def bound_forward_pass(
    param_l: list[torch.Tensor], param_u: list[torch.Tensor], x0_l: torch.Tensor, x0_u: torch.Tensor, **kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using the tightest of the double interval bound propagation and the
    double-interval crown algorithms

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
    """
    # compute both the crown and ibp bound
    activations_ibp_l, activations_ibp_u = ibp.bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs)
    activations_crown_l, activations_crown_u = crown.bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs)
    # compute the highest lower bound and lowest upper bound
    activations_l, activations_u = [], []
    for i in range(len(activations_ibp_l)):
        activations_l.append(torch.max(activations_ibp_l[i], activations_crown_l[i]))
        activations_u.append(torch.min(activations_ibp_u[i], activations_crown_u[i]))

    return activations_l, activations_u
