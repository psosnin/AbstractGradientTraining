import torch
import torch.nn.functional as F

"""
Functions that perform the nominal forward and backward passes through the network.
"""


def nominal_forward_pass(x0, params):
    """
    Perform the forward pass through the network with nominal parameters.
    """
    assert len(x0.shape) == 3  # this function expects a batched input
    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in params]
    x = [x0]
    for (Wk, bk) in zip(params[::2], params[1::2]):
        xhat = Wk @ x[-1] + bk
        x.append(F.relu(xhat))
    x.pop()  # no relu for last layer
    return xhat, x


def nominal_backward_pass(dL, params, inter):
    """
    Perform the backward pass through the network with nominal parameters given dL is the first partial derivative of
    the loss with respect to the logits.
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
