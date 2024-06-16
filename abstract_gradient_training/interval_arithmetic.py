import torch
import torch.nn.functional as F
from abstract_gradient_training import bound_utils

"""
Interval Helper Functions
"""


def propagate_affine(x_l, x_u, W_l, W_u, b_l, b_u):
    """
    Compute an interval bound on the affine transformation A @ x + b.
    """
    bound_utils.validate_interval(b_l, b_u)
    h_l, h_u = propagate_matmul(W_l, W_u, x_l, x_u)
    return h_l + b_l, h_u + b_u


def propagate_matmul(A_l, A_u, B_l, B_u):
    """
    Compute an interval bound on the matrix multiplication A @ B using Rump's algorithm.
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
    A_mu = (A_u + A_l) / 2
    A_r = (A_u - A_l) / 2
    B_mu = (B_u + B_l) / 2
    B_r = (B_u - B_l) / 2

    H_mu = A_mu @ B_mu
    H_r = torch.abs(A_mu) @ B_r + A_r @ torch.abs(B_mu) + A_r @ B_r
    H_l = H_mu - H_r
    H_u = H_mu + H_r

    bound_utils.validate_interval(H_l, H_u)
    return H_l, H_u


def propagate_elementwise(A_l, A_u, B_l, B_u):
    """
    Compute an interval bound on the matrix multiplication A * B.
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
    cases = torch.stack([A_l * B_l, A_u * B_l, A_l * B_u, A_u * B_u])
    lb = torch.min(cases, dim=0)[0]
    ub = torch.max(cases, dim=0)[0]
    bound_utils.validate_interval(lb, ub)
    return lb, ub


def propagate_conv2d(x_l, x_u, W, b, stride=1, padding=0):
    """
    Propagate the interval over x_l and x_u through the convolutional layer with fixed weights and biases.
    """
    # validate shapes and bounds
    bound_utils.validate_interval(x_l, x_u)
    assert x_l.dim() == 4
    assert W.dim() == 4
    assert b.dim() == 1  # allow the bias to be a vector, even though for affine layers we require it to be at least 2d

    x_mu = (x_u + x_l) / 2
    x_r = (x_u - x_l) / 2

    H_mu = F.conv2d(x_mu, W, bias=None, stride=stride, padding=padding)
    H_r = F.conv2d(x_r, torch.abs(W), bias=None, stride=stride, padding=padding)

    H_l = H_mu - H_r
    H_u = H_mu + H_r
    bound_utils.validate_interval(H_l, H_u)
    return H_l + b.view(1, -1, 1, 1), H_u + b.view(1, -1, 1, 1)


def propagate_tanh(x_l, x_u):
    """
    Compute an interval bound on the tanh function.
    """
    bound_utils.validate_interval(x_l, x_u)
    return F.tanh(x_l), F.tanh(x_u)


def propagate_relu(A_l, A_u):
    """
    Compute an interval bound on the ReLU function.
    """
    bound_utils.validate_interval(A_l, A_u)
    return F.relu(A_l), F.relu(A_u)


def propagate_softmax(A_l, A_u):
    """
    Compute an interval bound on the softmax function.
    """
    bound_utils.validate_interval(A_l, A_u)
    if len(A_l.shape) != 3:
        raise NotImplementedError(f"Only batched input supported for propagate softmax")
    I = torch.eye(A_l.shape[-2]).type(A_l.dtype).unsqueeze(0).to(A_l.device)
    # calculate bounds on the post-softmax output by choosing the best and worst case logits for each class output.
    y_l = torch.diagonal(torch.nn.Softmax(dim=-2)((I * A_l + (1 - I) * A_u)), dim1=-2, dim2=-1).unsqueeze(-1)
    y_u = torch.diagonal(torch.nn.Softmax(dim=-2)((I * A_u + (1 - I) * A_l)), dim1=-2, dim2=-1).unsqueeze(-1)
    return y_l, y_u


def union(A_l, A_u, B_l, B_u):
    """
    Compute the elementwise union of the two matrix intervals A and B.
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
    return torch.minimum(A_l, B_l), torch.maximum(A_u, B_u)
