"""Bounds that combine the double interval bound propagation and the double-interval crown algorithms."""

import torch

from abstract_gradient_training.bounds import interval_bound_propagation as ibp
from abstract_gradient_training.bounds import crown


def bound_forward_pass(
    param_l: list[torch.Tensor], param_u: list[torch.Tensor], x0_l: torch.Tensor, x0_u: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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
        logit_l (torch.Tensor): lower bounds on the logits
        logit_u (torch.Tensor): upper bounds on the logits
        inter_l (list[torch.Tensor]): list of lower bounds on the (post-relu) intermediate activations [x0, ..., xL-1]
        inter_u (list[torch.Tensor]): list of upper bounds on the (post-relu) intermediate activations [x0, ..., xL-1]
    """
    # compute both the crown and ibp bound
    logit_ibp_l, logit_ibp_u, inter_ibp_l, inter_ibp_u = ibp.bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs)
    logit_crown_l, logit_crown_u, inter_crown_l, inter_crown_u = crown.bound_forward_pass(
        param_l, param_u, x0_l, x0_u, **kwargs
    )
    # compute the highest lower bound and lowest upper bound
    logit_l = torch.max(logit_ibp_l, logit_crown_l)
    logit_u = torch.min(logit_ibp_u, logit_crown_u)
    inter_l, inter_u = [], []
    for i in range(len(inter_ibp_l)):
        inter_l.append(torch.max(inter_ibp_l[i], inter_crown_l[i]))
        inter_u.append(torch.min(inter_ibp_u[i], inter_crown_u[i]))

    return logit_l, logit_u, inter_l, inter_u
