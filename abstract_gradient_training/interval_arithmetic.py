"""Interval arithmetic helper functions"""

import inspect
import logging

import torch
import torch.nn.functional as F


def propagate_affine(
    x_l: torch.Tensor, x_u: torch.Tensor, W_l: torch.Tensor, W_u: torch.Tensor, b_l: torch.Tensor, b_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the affine transformation A @ x + b.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor x.
        x_u (torch.Tensor): Upper bound of the input tensor x.
        W_l (torch.Tensor): Lower bound of the weight matrix A.
        W_u (torch.Tensor): Upper bound of the weight matrix A.
        b_l (torch.Tensor): Lower bound of the bias vector b.
        b_u (torch.Tensor): Upper bound of the bias vector b.

    Returns:
        e_l (torch.Tensor): Lower bound of the output tensor.
        e_u (torch.Tensor): Upper bound of the output tensor.
    """
    validate_interval(b_l, b_u)
    h_l, h_u = propagate_matmul(W_l, W_u, x_l, x_u)
    return h_l + b_l, h_u + b_u


def propagate_matmul(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the matrix multiplication A @ B using Rump's algorithm.

    Args:
        A_l (torch.Tensor): Lower bound of the matrix A.
        A_u (torch.Tensor): Upper bound of the matrix A.
        B_l (torch.Tensor): Lower bound of the matrix B.
        B_u (torch.Tensor): Upper bound of the matrix B.

    Returns:
        H_l (torch.Tensor): Lower bound of the output tensor.
        H_u (torch.Tensor): Upper bound of the output tensor.
    """
    validate_interval(A_l, A_u)
    validate_interval(B_l, B_u)
    A_mu = (A_u + A_l) / 2
    A_r = (A_u - A_l) / 2
    B_mu = (B_u + B_l) / 2
    B_r = (B_u - B_l) / 2

    H_mu = A_mu @ B_mu
    H_r = torch.abs(A_mu) @ B_r + A_r @ torch.abs(B_mu) + A_r @ B_r
    H_l = H_mu - H_r
    H_u = H_mu + H_r

    validate_interval(H_l, H_u)
    return H_l, H_u


def propagate_elementwise(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the matrix multiplication A * B.

    Args:
        A_l (torch.Tensor): Lower bound of the matrix A.
        A_u (torch.Tensor): Upper bound of the matrix A.
        B_l (torch.Tensor): Lower bound of the matrix B.
        B_u (torch.Tensor): Upper bound of the matrix B.

    Returns:
        H_l (torch.Tensor): Lower bound of the output tensor.
        H_u (torch.Tensor): Upper bound of the output tensor.
    """
    validate_interval(A_l, A_u)
    validate_interval(B_l, B_u)
    cases = torch.stack([A_l * B_l, A_u * B_l, A_l * B_u, A_u * B_u])
    lb = torch.min(cases, dim=0)[0]
    ub = torch.max(cases, dim=0)[0]
    validate_interval(lb, ub)
    return lb, ub


def propagate_conv2d(
    x_l: torch.Tensor, x_u: torch.Tensor, W: torch.Tensor, b: torch.Tensor, stride: int = 1, padding: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate the interval over x_l and x_u through the convolutional layer with fixed weights and biases.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor x.
        x_u (torch.Tensor): Upper bound of the input tensor x.
        W (torch.Tensor): Weight tensor of the convolutional layer.
        b (torch.Tensor): Bias tensor of the convolutional layer.
        stride (int, optional): Stride of the convolutional layer. Defaults to 1.
        padding (int, optional): Padding of the convolutional layer. Defaults to 0.

    Returns:
        e_l (torch.Tensor): Lower bound of the output tensor.
        e_u (torch.Tensor): Upper bound of the output tensor.
    """

    # validate shapes and bounds
    validate_interval(x_l, x_u)
    assert x_l.dim() == 4
    assert W.dim() == 4
    assert b.dim() == 1  # allow the bias to be a vector, even though for affine layers we require it to be at least 2d

    x_mu = (x_u + x_l) / 2
    x_r = (x_u - x_l) / 2

    H_mu = F.conv2d(x_mu, W, bias=None, stride=stride, padding=padding)
    H_r = F.conv2d(x_r, torch.abs(W), bias=None, stride=stride, padding=padding)

    H_l = H_mu - H_r
    H_u = H_mu + H_r
    validate_interval(H_l, H_u)
    return H_l + b.view(1, -1, 1, 1), H_u + b.view(1, -1, 1, 1)


def propagate_softmax(A_l: torch.Tensor, A_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the softmax function given an interval over the input logits.

    Args:
        A_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound of the input logits.
        A_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound of the input logits.

    Returns:
        y_l (torch.Tensor): [batchsize x output_dim x 1] Lower bound of the output probabilities.
        y_u (torch.Tensor): [batchsize x output_dim x 1] Upper bound of the output probabilities.
    """
    validate_interval(A_l, A_u)
    if len(A_l.shape) != 3:
        raise NotImplementedError("Only batched input supported for propagate softmax")
    I = torch.eye(A_l.shape[-2]).type(A_l.dtype).unsqueeze(0).to(A_l.device)
    # calculate bounds on the post-softmax output by choosing the best and worst case logits for each class output.
    y_l = torch.diagonal(torch.nn.Softmax(dim=-2)((I * A_l + (1 - I) * A_u)), dim1=-2, dim2=-1).unsqueeze(-1)
    y_u = torch.diagonal(torch.nn.Softmax(dim=-2)((I * A_u + (1 - I) * A_l)), dim1=-2, dim2=-1).unsqueeze(-1)
    return y_l, y_u


def validate_interval(l: torch.Tensor, u: torch.Tensor) -> None:
    """
    Validate an arbitrary interval [l, u] and log any violations of the bound at a level based on the size of the
    violation.

    Args:
        l (torch.Tensor): Lower bound of the interval.
        u (torch.Tensor): Upper bound of the interval.
    """
    diff = torch.max(l - u)  # this should be negative
    if diff <= 0:
        return
    func_name = inspect.currentframe().f_back.f_code.co_name
    if diff > 1e-3:  # a major infraction of the bound
        logging.error("Violated bound in %s: %s", func_name, diff)
    elif diff > 1e-4:  # a minor infraction of the bound
        logging.warning("Violated bound in %s: %s", func_name, diff)
    elif diff > 1e-5:  # a minor infraction of the bound
        logging.info("Violated bound in %s: %s", func_name, diff)
    elif diff > 0:  # a tiny infraction of the bound
        logging.debug("Violated bound in %s: %s", func_name, diff)
