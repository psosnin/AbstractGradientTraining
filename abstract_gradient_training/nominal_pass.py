"""Functions that perform the nominal forward and backward passes through the network."""

import torch
import torch.nn.functional as F


def nominal_forward_pass(x0: torch.Tensor, params: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Perform the forward pass through the network with the given nominal parameters.

    Args:
        x0 (torch.Tensor): [batchsize x input_dim x 1] tensor of inputs to the network
        params (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].

    Returns:
        xhat (torch.Tensor): [batchsize x output_dim x 1] tensor of logits
        x (list[torch.Tensor]): List of the intermediate activations of the network [x0, x1, ..., xn].
    """

    assert len(x0.shape) == 3  # this function expects a batched input
    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in params]
    x = [x0]
    for wk, bk in zip(params[::2], params[1::2]):
        xhat = wk @ x[-1] + bk
        x.append(F.relu(xhat))
    x.pop()  # no relu for last layer
    return xhat, x


def nominal_backward_pass(
    dL: torch.Tensor, params: list[torch.Tensor], inter: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Perform the backward pass through the network with nominal parameters given dL is the first partial derivative of
    the loss with respect to the logits.

    Args:
        dL (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the gradient of the loss with
                           respect to the logits of the network.
        params (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        inter (list[torch.Tensor]): List of the intermediate activations of the network [x0, x1, ..., xn].
    Returns:
        list[torch.Tensor]: List of gradients of the network [dW1, db1, ..., dWm, dbm]
    """

    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in params]
    dL = dL if len(dL.shape) == 3 else dL.unsqueeze(-1)
    W = params[::2]
    dW = dL @ inter[-1].transpose(-2, -1)
    grads = [dL, dW]

    # compute gradients for each layer
    for i in range(len(W) - 1, 0, -1):
        dL = W[i].T @ dL
        dL = dL * torch.sign(inter[i])
        dW = dL @ inter[i - 1].transpose(-2, -1)
        grads.append(dL)
        grads.append(dW)

    grads.reverse()
    return grads
