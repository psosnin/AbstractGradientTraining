import torch

from abstract_gradient_training.bounds import interval_bound_propagation as ibp
from abstract_gradient_training.bounds import crown


def bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs):
    """
    Compute both the crown bounds and the interval bounds and return the elementwise minimum and maximum of the two.
    Parameters:
        param_l: lower bounds on the weights and biases
        param_u: upper bounds on the weights and biases
        x0_l: [batchsize x input_dim x 1] lower bound on the input batch to the network
        x0_u: [batchsize x input_dim x 1] upper bound on the input batch to the network
    Returns:
        logit_l: lower bounds on the logits
        logit_u: upper bounds on the logits
        inter_l: list of lower bounds on the intermediate activations (input interval, then post-relu bounds)
        inter_u: list of upper bounds on the intermediate activations (input interval, then post-relu bounds)
    """
    # compute both the crown and ibp bound
    logit_ibp_l, logit_ibp_u, inter_ibp_l, inter_ibp_u = ibp.bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs)
    logit_crown_l, logit_crown_u, inter_crown_l, inter_crown_u = crown.bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs)
    # compute the highest lower bound and lowest upper bound
    logit_l = torch.max(logit_ibp_l, logit_crown_l)
    logit_u = torch.min(logit_ibp_u, logit_crown_u)
    inter_l, inter_u = [], []
    for i in range(len(inter_ibp_l)):
        inter_l.append(torch.max(inter_ibp_l[i], inter_crown_l[i]))
        inter_u.append(torch.min(inter_ibp_u[i], inter_crown_u[i]))

    return logit_l, logit_u, inter_l, inter_u
