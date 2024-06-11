import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training import bound_utils


def bound_forward_pass(param_l, param_u, x0_l, x0_u, **kwargs):
    """
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
    # validate the input
    param_l, param_u, h_l, h_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
    

    W_l, b_l = param_l[::2], param_l[1::2]
    W_u, b_u = param_u[::2], param_u[1::2]
    inter_l, inter_u = [h_l], [h_u]  # containers to hold intermediate bounds

    # propagate interval bound through each layer
    for i in range(len(W_l)):
        h_l, h_u = interval_arithmetic.propagate_affine(inter_l[-1], inter_u[-1], W_l[i], W_u[i], b_l[i], b_u[i])
        if i < len(W_l) - 1:
            inter_l.append(torch.nn.ReLU()(h_l))
            inter_u.append(torch.nn.ReLU()(h_u))

    return h_l, h_u, inter_l, inter_u


def bound_backward_pass(dL_min, dL_max, param_l, param_u, inter_l, inter_u, **kwargs):
    """
    Perform a backward pass through the network to compute bounds on the gradients of the loss.
    Parameters:
        dL_min: lower bound on the gradient of the loss with respect to the logits
        dL_max: upper bound on the gradient of the loss with respect to the logits
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

    # get weight matrix bounds
    W_l, W_u = param_l[::2], param_u[::2]

    # compute the gradient of the loss with respect to the weights and biases of the last layer
    dW_min, dW_max = interval_arithmetic.propagate_matmul(
        dL_min, dL_max, inter_l[-1].transpose(-2, -1), inter_u[-1].transpose(-2, -1)
    )

    grads_l, grads_u = [dL_min, dW_min], [dL_max, dW_max]

    # compute gradients for each layer
    for i in range(len(W_l) - 1, 0, -1):
        dL_dz_min, dL_dz_max = interval_arithmetic.propagate_matmul(W_l[i].T, W_u[i].T, dL_min, dL_max)
        min_inter = (inter_l[i] > 0).type(inter_l[i].dtype)
        max_inter = (inter_u[i] > 0).type(inter_u[i].dtype)
        dL_min, dL_max = interval_arithmetic.propagate_elementwise(dL_dz_min, dL_dz_max, min_inter, max_inter)
        dW_min, dW_max = interval_arithmetic.propagate_matmul(
            dL_min, dL_max, inter_l[i - 1].transpose(-2, -1), inter_u[i - 1].transpose(-2, -1)
        )

        grads_l.append(dL_min), grads_l.append(dW_min)
        grads_u.append(dL_max), grads_u.append(dW_max)

    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u

