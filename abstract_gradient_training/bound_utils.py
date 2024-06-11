import logging
import inspect
import torch


def validate_forward_bound_input(param_l, param_u, x0_l, x0_u):
    """
    Validate and reshape input arguments to forward bounding functions.
    The bounding functions require the following shapes:
        Wi: [n x m] tensor
        bi: [n x 1] tensor
        x0: [batchsize x input_dim x 1] tensor
    Parameters:
        param_l: list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u: list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l: lower bound on the input batch to the network
        x0_u: upper bound on the input batch to the network
    """
    # check the shapes of the parameters and perform the necessary reshaping
    assert all([p.dim() == 2 for p in param_l[::2]]), "Weights must be 2D tensors"
    assert all([p.dim() in [1, 2] for p in param_l[1::2]]), "Biases must be 1D or 2D tensors"
    assert all([p_l.dim() == p_u.dim() for p_l, p_u in zip(param_l, param_u)]), "Bounds must have the same shape"
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


def validate_backward_bound_input(dL_min, dL_max, param_l, param_u, inter_l, inter_u):
    """
    Validate and reshape input arguments to forward bounding functions.
    The bounding functions require the following shapes:
        Wi: [n x m] tensor
        bi: [n x 1] tensor
        dL: [batchsize x output_dim x 1] tensor
        inter[i]: [batchsize x hidden_dim x 1] tensors
    Parameters:
        dL_min: [batchsize x output_dim x 1] lower bound on the gradient of the loss with respect to the logits
        dL_max: [batchsize x output_dim x 1] upper bound on the gradient of the loss with respect to the logits
        param_l: lower bounds on the weights and biases given as a list [W1, b1, ..., Wm, bm]
        param_u: upper bounds on the weights and biases given as a list [W1, b1, ..., Wm, bm]
        inter_l: lower bounds on the intermediate activations given as a list [x0, ..., xL]
        inter_u: upper bounds on the intermediate activations given as a list [x0, ..., xL]
    """
    # check the shapes of the parameters and perform the necessary reshaping
    assert all([p.dim() == 2 for p in param_l[::2]]), "Weights must be 2D tensors"
    assert all([p.dim() in [1, 2] for p in param_l[1::2]]), "Biases must be 1D or 2D tensors"
    assert all([p_l.dim() == p_u.dim() for p_l, p_u in zip(param_l, param_u)]), "Bounds must have the same shape"
    param_l = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in param_l]
    param_u = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in param_u]

    # check the shapes of the gradient
    assert dL_min.dim() == 3, "First gradient of the loss must have shape [batchsize x output_dim x 1]"
    assert dL_max.dim() == 3, "First gradient of the loss must have shape [batchsize x output_dim x 1]"

    # check the shapes of the intermediate bounds
    assert all([x_l.dim() == 3 for x_l in inter_l]), "Intermediate bounds must have shape [batchsize x hidden_dim x 1]"
    assert all([x_u.dim() == 3 for x_u in inter_u]), "Intermediate bounds must have shape [batchsize x hidden_dim x 1]"

    return dL_min, dL_max, param_l, param_u, inter_l, inter_u


def validate_interval(l, u):
    """
    Validate an arbitrary interval [l, u] and log any violations of the bound at a level based on the size of the
    violation.
    """
    diff = torch.max(l - u)  # this should be negative
    if diff <= 0:
        return
    func_name = inspect.currentframe().f_back.f_code.co_name
    if diff > 1e-3:  # a major infraction of the bound
        logging.error(f"Violated bound in {func_name}: {diff}")
    elif diff > 1e-4:  # a minor infraction of the bound
        logging.warning(f"Violated bound in {func_name}: {diff}")
    elif diff > 0:  # a tiny infraction of the bound
        logging.info(f"Violated bound in {func_name}: {diff}")
