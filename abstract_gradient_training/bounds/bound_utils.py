"""Functions for validating the inputs into the bounding functions."""

import torch


def validate_forward_bound_input(
    param_l: list[torch.Tensor], param_u: list[torch.Tensor], x0_l: torch.Tensor, x0_u: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Validate and reshape input arguments to forward bounding functions.
    The bounding functions require the following shapes:
        Wi: [n x m] tensor
        bi: [n x 1] tensor
        x0: [batchsize x input_dim x 1] tensor

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): lower bound on the input batch to the network
        x0_u (torch.Tensor): upper bound on the input batch to the network
    """
    # check the shapes of the parameters and perform the necessary reshaping
    assert all(p.dim() == 2 for p in param_l[::2]), "Weights must be 2D tensors"
    assert all(p.dim() in [1, 2] for p in param_l[1::2]), "Biases must be 1D or 2D tensors"
    assert all(p_l.dim() == p_u.dim() for p_l, p_u in zip(param_l, param_u)), "Bounds must have the same shape"
    param_l = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in param_l]
    param_u = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in param_u]

    # check datatypes (pytorch cuda errors if types don't match)
    assert param_l[0].dtype == x0_l.dtype, "Parameter and input must have the same datatype"

    # check the shapes of the input x0_l and x0_u
    # we don't reshape the input bounds as there are cases where we cannot infer the difference between
    # an input of [batchsize x input_dim] and [input_dim_1 x input_dim_2]
    assert x0_l.dim() == 3, "Bounding function expects batched input with shape [batchsize x input_dim x 1]"
    assert x0_u.dim() == 3, "Bounding function expects batched input with shape [batchsize x input_dim x 1]"

    return param_l, param_u, x0_l, x0_u


def validate_backward_bound_input(
    dL_min: torch.Tensor,
    dL_max: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    activations_l: list[torch.Tensor],
    activations_u: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Validate and reshape input arguments to forward bounding functions.
    The bounding functions require the following shapes:
        Wi: [n x m] tensor
        bi: [n x 1] tensor
        dL: [batchsize x output_dim x 1] tensor
        activations[i]: [batchsize x hidden_dim x 1] tensors

    Args:
        dL_min (torch.Tensor): [batchsize x output_dim x 1] lower bound on the loss gradient with respect to the logits
        dL_max (torch.Tensor): [batchsize x output_dim x 1] upper bound on the loss gradient with respect to the logits
        param_l (list[torch.Tensor]): lower bounds on the weights and biases given as a list [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): upper bounds on the weights and biases given as a list [W1, b1, ..., Wm, bm]
        activations_l (list[torch.Tensor]): list of lower bounds on all (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on all (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor has shape [batchsize x dim x 1].
    """
    # check the shapes of the parameters and perform the necessary reshaping
    assert all(p.dim() == 2 for p in param_l[::2]), "Weights must be 2D tensors"
    assert all(p.dim() in [1, 2] for p in param_l[1::2]), "Biases must be 1D or 2D tensors"
    assert all(p_l.dim() == p_u.dim() for p_l, p_u in zip(param_l, param_u)), "Bounds must have the same shape"
    param_l = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in param_l]
    param_u = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in param_u]

    # check the shapes of the gradient
    assert dL_min.dim() == 3, "First gradient of the loss must have shape [batchsize x output_dim x 1]"
    assert dL_max.dim() == 3, "First gradient of the loss must have shape [batchsize x output_dim x 1]"

    # check the shapes of the intermediate bounds
    assert all(x_l.dim() == 3 for x_l in activations_l), "Activation bounds must have shape [batchsize x dim x 1]"
    assert all(x_u.dim() == 3 for x_u in activations_u), "Activation bounds must have shape [batchsize x dim x 1]"

    return dL_min, dL_max, param_l, param_u, activations_l, activations_u
