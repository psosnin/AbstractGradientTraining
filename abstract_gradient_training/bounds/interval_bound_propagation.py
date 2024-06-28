"""Interval bound propagation."""

import torch
import torch.nn.functional as F

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounds import bound_utils


def bound_forward_pass(
    param_l: list[torch.Tensor], param_u: list[torch.Tensor], x0_l: torch.Tensor, x0_u: torch.Tensor, **kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using double interval bound propagation.

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
    """

    # validate the input
    param_l, param_u, h_l, h_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
    W_l, b_l = param_l[::2], param_l[1::2]
    W_u, b_u = param_u[::2], param_u[1::2]
    activations_l, activations_u = [h_l], [h_u]  # containers to hold intermediate bounds

    # propagate interval bound through each layer
    for i in range(len(W_l)):
        if i == 0:  # no relu for input layer
            h_l, h_u = interval_arithmetic.propagate_affine(x0_l, x0_u, W_l[i], W_u[i], b_l[i], b_u[i])
        else:
            h_l, h_u = interval_arithmetic.propagate_affine(
                F.relu(activations_l[-1]), F.relu(activations_u[-1]), W_l[i], W_u[i], b_l[i], b_u[i]
            )
        activations_l.append(h_l)
        activations_u.append(h_u)

    return activations_l, activations_u


def bound_backward_pass(
    dL_min: torch.Tensor,
    dL_max: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    activations_l: list[torch.Tensor],
    activations_u: list[torch.Tensor],
    **kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters, intermediate activations and the first partial derivative of the loss, compute
    bounds on the gradients of the loss with respect to the parameters of the network using double interval bound
    propagation.

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

    # get weight matrix bounds
    W_l, W_u = param_l[::2], param_u[::2]

    # compute the gradient of the loss with respect to the weights and biases of the last layer
    dW_min, dW_max = interval_arithmetic.propagate_matmul(
        dL_min, dL_max, activations_l[-2].transpose(-2, -1), activations_u[-2].transpose(-2, -1)
    )

    grads_l, grads_u = [dL_min, dW_min], [dL_max, dW_max]

    # compute gradients for each layer
    for i in range(len(W_l) - 1, 0, -1):
        dL_dz_min, dL_dz_max = interval_arithmetic.propagate_matmul(W_l[i].T, W_u[i].T, dL_min, dL_max)
        min_act = (activations_l[i] > 0).type(activations_l[i].dtype)
        max_act = (activations_u[i] > 0).type(activations_u[i].dtype)
        dL_min, dL_max = interval_arithmetic.propagate_elementwise(dL_dz_min, dL_dz_max, min_act, max_act)
        dW_min, dW_max = interval_arithmetic.propagate_matmul(
            dL_min, dL_max, activations_l[i - 1].transpose(-2, -1), activations_u[i - 1].transpose(-2, -1)
        )
        grads_l.append(dL_min)
        grads_l.append(dW_min)
        grads_u.append(dL_max)
        grads_u.append(dW_max)

    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u
