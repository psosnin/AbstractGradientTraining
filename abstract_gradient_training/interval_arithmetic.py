"""Interval arithmetic helper functions"""

import inspect
import logging
from typing import Literal
from collections.abc import Callable

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


def propagate_affine(
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    W_l: torch.Tensor,
    W_u: torch.Tensor,
    b_l: torch.Tensor,
    b_u: torch.Tensor,
    interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the affine transformation A @ x + b using Rump's algorithm.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor x.
        x_u (torch.Tensor): Upper bound of the input tensor x.
        W_l (torch.Tensor): Lower bound of the weight matrix A.
        W_u (torch.Tensor): Upper bound of the weight matrix A.
        b_l (torch.Tensor): Lower bound of the bias vector b.
        b_u (torch.Tensor): Upper bound of the bias vector b.
        interval_matmul (str): Method to use for interval matmul, one of ["rump", "exact", "nguyen"].

    Returns:
        e_l (torch.Tensor): Lower bound of the output tensor.
        e_u (torch.Tensor): Upper bound of the output tensor.
    """
    validate_interval(b_l, b_u)
    h_l, h_u = propagate_matmul(W_l, W_u, x_l, x_u, interval_matmul)
    return h_l + b_l, h_u + b_u


def propagate_matmul(
    A_l: torch.Tensor,
    A_u: torch.Tensor,
    B_l: torch.Tensor,
    B_u: torch.Tensor,
    interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the matrix multiplication A @ B using the specified method.

    Args:
        A_l (torch.Tensor): Lower bound of the matrix A.
        A_u (torch.Tensor): Upper bound of the matrix A.
        B_l (torch.Tensor): Lower bound of the matrix B.
        B_u (torch.Tensor): Upper bound of the matrix B.
        interval_matmul (str): Method to use for interval matmul, one of ["rump", "exact", "nguyen"].

    Returns:
        H_l (torch.Tensor): Lower bound of the output tensor.
        H_u (torch.Tensor): Upper bound of the output tensor.
    """
    validate_interval(A_l, A_u)
    validate_interval(B_l, B_u)
    if interval_matmul == "rump":
        H_l, H_u = propagate_matmul_rump(A_l, A_u, B_l, B_u)
    elif interval_matmul == "exact":
        H_l, H_u = propagate_matmul_exact(A_l, A_u, B_l, B_u)
    elif interval_matmul == "nguyen":
        H_l, H_u = propagate_matmul_nguyen(A_l, A_u, B_l, B_u)
    else:
        raise ValueError(f"Unknown interval matmul method: {interval_matmul}")
    validate_interval(H_l, H_u)
    return H_l, H_u


def propagate_matmul_rump(
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
    A_mu = (A_u + A_l) / 2
    A_r = (A_u - A_l) / 2
    B_mu = (B_u + B_l) / 2
    B_r = (B_u - B_l) / 2

    H_mu = A_mu @ B_mu
    H_r = torch.abs(A_mu) @ B_r + A_r @ torch.abs(B_mu) + A_r @ B_r
    H_l = H_mu - H_r
    H_u = H_mu + H_r
    return H_l, H_u


def propagate_matmul_nguyen(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the matrix multiplication A @ B using Nguyen's algorithm. The algorithm performs
    a decomposition of A into two intervals A^0 and A^* such that
    - A^0 is a zero-centered interval
    - A^* is an interval that does not contain zero

    Args:
        A_l (torch.Tensor): Lower bound of the matrix A.
        A_u (torch.Tensor): Upper bound of the matrix A.
        B_l (torch.Tensor): Lower bound of the matrix B.
        B_u (torch.Tensor): Upper bound of the matrix B.

    Returns:
        H_l (torch.Tensor): Lower bound of the output tensor.
        H_u (torch.Tensor): Upper bound of the output tensor.
    """
    # A0 is an interval centered at zero
    A0_l, A0_u = torch.zeros_like(A_l), torch.zeros_like(A_l)
    # Astar is an interval not containing zero.
    # we further split Astar into its negative and positive parts
    Astar_neg_l, Astar_neg_u = torch.zeros_like(A_l), torch.zeros_like(A_l)
    Astar_pos_l, Astar_pos_u = torch.zeros_like(A_l), torch.zeros_like(A_l)

    # case where A_l * A_u >= 0
    condition = (A_l >= 0) & (A_u >= 0)
    Astar_pos_l = torch.where(condition, A_l, Astar_pos_l)
    Astar_pos_u = torch.where(condition, A_u, Astar_pos_u)
    condition = (A_l <= 0) & (A_u <= 0)
    Astar_neg_l = torch.where(condition, A_l, Astar_neg_l)
    Astar_neg_u = torch.where(condition, A_u, Astar_neg_u)

    # case where A_l < 0 < |A_l| <= A_u
    condition = (A_l < 0) & (torch.abs(A_l) <= A_u)
    A0_l = torch.where(condition, A_l, A0_l)
    A0_u = torch.where(condition, -A_l, A0_u)
    Astar_pos_u = torch.where(condition, A_l + A_u, Astar_pos_u)

    # case where A_l < 0 < A_u < |A_l|
    condition = (A_l < 0) & (0 < A_u) & (A_u < torch.abs(A_l))
    A0_l = torch.where(condition, -A_u, A0_l)
    A0_u = torch.where(condition, A_u, A0_u)
    Astar_neg_l = torch.where(condition, A_l + A_u, Astar_neg_l)

    assert torch.allclose(A_l, A0_l + Astar_pos_l + Astar_neg_l)
    assert torch.allclose(A_u, A0_u + Astar_pos_u + Astar_neg_u)
    assert torch.allclose(A0_l + A0_u, torch.zeros_like(A0_l))
    assert torch.all((Astar_neg_l <= 0) & (Astar_neg_u <= 0))
    assert torch.all((Astar_pos_l >= 0) & (Astar_pos_u >= 0))

    # compute the interval A0 @ B
    H_u = A0_u @ torch.maximum(torch.abs(B_l), torch.abs(B_u))
    H_l = -H_u

    # split the matrix B into its negative and positive parts
    B_neg_l, B_neg_u = torch.clamp(B_l, max=0), torch.clamp(B_u, max=0)
    B_pos_l, B_pos_u = torch.clamp(B_l, min=0), torch.clamp(B_u, min=0)

    # compute the interval Astar_pos @ B
    H_l += Astar_pos_l @ B_pos_l + Astar_pos_u @ B_neg_l
    H_u += Astar_pos_u @ B_pos_u + Astar_pos_l @ B_neg_u

    # compute the interval Astar_neg @ B
    H_l += Astar_neg_l @ B_pos_u + Astar_neg_u @ B_neg_u
    H_u += Astar_neg_u @ B_pos_l + Astar_neg_l @ B_neg_l

    return H_l, H_u


def propagate_matmul_exact(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the matrix multiplication A @ B using exact interval arithmetic.

    Args:
        A_l (torch.Tensor): Lower bound of the matrix A.
        A_u (torch.Tensor): Upper bound of the matrix A.
        B_l (torch.Tensor): Lower bound of the matrix B.
        B_u (torch.Tensor): Upper bound of the matrix B.

    Returns:
        H_l (torch.Tensor): Lower bound of the output tensor.
        H_u (torch.Tensor): Upper bound of the output tensor.
    """
    E_l, E_u = propagate_elementwise(A_l.unsqueeze(-1), A_u.unsqueeze(-1), B_l.unsqueeze(-3), B_u.unsqueeze(-3))
    return E_l.sum(-2), E_u.sum(-2)


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


def propagate_norm(x_l: torch.Tensor, x_u: torch.Tensor, p: float = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an interval bound on the Lp norm of the input interval [x_l, x_u].
    If dim x > 1, then the first dimension is treated as a batch dimension and the norm is computed over the remaining
    dimensions flattened into a single dimension.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor x.
        x_u (torch.Tensor): Upper bound of the input tensor x.
        p (float, optional): Order of the norm. Defaults to 2.

    Returns:
        e_l (torch.Tensor): Lower bound of the output tensor.
        e_u (torch.Tensor): Upper bound of the output tensor.
    """
    validate_interval(x_l, x_u)
    if x_l.dim() > 2:
        x_l, x_u = x_l.flatten(start_dim=1), x_u.flatten(start_dim=1)

    if p == 1:
        # compute interval over abs(x)
        x_abs_l = torch.minimum(x_l.abs(), x_u.abs()) * ((x_l > 0) | (x_u < 0))
        x_abs_u = torch.maximum(x_l.abs(), x_u.abs())
        validate_interval(x_abs_l, x_abs_u)
        norm_l = x_abs_l.sum(-1)
        norm_u = x_abs_u.sum(-1)
    elif p == 2:
        # compute interval over x^2, then sum and take the sqrt
        x_square_l = torch.minimum(x_l.abs(), x_u.abs()).square() * ((x_l > 0) | (x_u < 0))
        x_square_u = torch.maximum(x_l.abs(), x_u.abs()).square()
        validate_interval(x_square_l, x_square_u)
        norm_l = x_square_l.sum(-1).sqrt()
        norm_u = x_square_u.sum(-1).sqrt()
    else:
        raise ValueError(f"Norm of order {p} not suppported.")
    validate_interval(norm_l, norm_u)
    return norm_l, norm_u


def propagate_conv2d(
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    W_l: torch.Tensor,
    W_u: torch.Tensor,
    b_l: torch.Tensor | None = None,
    b_u: torch.Tensor | None = None,
    transpose: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate the interval over x_l and x_u through the convolutional layer using Rump's algorithm.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor x.
        x_u (torch.Tensor): Upper bound of the input tensor x.
        W_l (torch.Tensor): Lower bound of the weight tensor of the convolutional layer.
        W_u (torch.Tensor): Upper bound of the weight tensor of the convolutional layer.
        b_l (torch.Tensor, optional): Lower bound of the bias tensor of the convolutional layer.
        b_u (torch.Tensor, optional): Upper bound of the bias tensor of the convolutional layer.
        transpose (bool, optional): Whether the convolution is a transposed convolution. Defaults to False.
        **kwargs: Additional arguments to pass to the convolutional layer.

    Returns:
        e_l (torch.Tensor): Lower bound of the output tensor.
        e_u (torch.Tensor): Upper bound of the output tensor.
    """

    # validate shapes and bounds
    validate_interval(x_l, x_u, msg="input interval")
    validate_interval(W_l, W_u, msg="weight interval")
    assert x_l.dim() == 4
    assert W_l.dim() == 4
    if b_l is not None:
        assert b_u is not None, "Upper bound of bias tensor must be provided if lower bound is provided"
        assert b_l.dim() == 1  # require the bias to be a vector, even though for affine layers we require it to be 2d

    # get the appropriate conv function to use
    def transform(x, W):
        if transpose:
            return F.conv_transpose2d(x, W, bias=None, **kwargs)
        return F.conv2d(x, W, bias=None, **kwargs)

    # apply the linear transform
    H_l, H_u = propagate_linear_transform(x_l, x_u, W_l, W_u, transform)

    # add the bias
    if b_l is not None and b_u is not None:
        validate_interval(b_l, b_u, msg="bias interval")
        H_l, H_u = H_l + b_l.view(1, -1, 1, 1), H_u + b_u.view(1, -1, 1, 1)
        validate_interval(H_l, H_u, msg="output interval with bias")

    return H_l, H_u


def propagate_linear_transform(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor, transform: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given any linear transformation f (i.e. f(A + B) = f(A) + f(B)), compute the interval bound on the output of the
    transformation given an interval over the input using Rump's algorithm.

    Args:
        A_l (torch.Tensor): Lower bound on the first input tensor A.
        A_u (torch.Tensor): Upper bound on the first input tensor A.
        B_l (torch.Tensor): Lower bound on the second input tensor B.
        B_u (torch.Tensor): Upper bound on the second input tensor B.
        transform (Callable): The linear transformation to apply to the input interval.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Interval over the output of the transformation.
    """
    # validate input
    validate_interval(A_l, A_u, msg="input interval A")
    validate_interval(B_l, B_u, msg="input interval B")

    # compute the "mean" and "radius" of the input intervals
    A_mu = (A_u + A_l) / 2
    A_r = (A_u - A_l) / 2
    B_mu = (B_u + B_l) / 2
    B_r = (B_u - B_l) / 2

    # compute the "mean" and "radius" of the output
    H_mu = transform(A_mu, B_mu)
    H_r = transform(torch.abs(A_mu), B_r) + transform(A_r, torch.abs(B_mu)) + transform(A_r, B_r)

    # convert to lower and upper bounds
    H_l = H_mu - H_r
    H_u = H_mu + H_r
    validate_interval(H_l, H_u, msg="output interval")
    return H_l, H_u


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


def validate_interval(l: torch.Tensor, u: torch.Tensor, n: torch.Tensor | None = None, msg: str = "") -> None:
    """
    Validate an arbitrary interval n in [l, u] and log any violations of the bound at a level based on the size of the
    violation.

    Args:
        l (torch.Tensor): Lower bound of the interval.
        u (torch.Tensor): Upper bound of the interval.
        n (torch.Tensor | None, optional): Nominal value of the interval. Defaults to None.
        msg (str, optional): Optional message to log with the bound violation for debugging purposes.
    """
    if n is None:
        diff = torch.max(l - u).item()
    else:
        diff = max(torch.max(l - u).item(), torch.max(l - n).item(), torch.max(n - u).item())
    # this should be negative if all bounds are satisfied
    if diff <= 0:
        return
    # get caller function name, if available
    current_frame = inspect.currentframe()
    f_back = current_frame.f_back if current_frame is not None else None
    func_name = f_back.f_code.co_name if f_back is not None else ""
    if diff > 1e-3:  # a major infraction of the bound
        LOGGER.error("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
    elif diff > 1e-4:  # a minor infraction of the bound
        LOGGER.warning("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
    elif diff > 1e-5:  # a minor infraction of the bound
        LOGGER.info("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
    elif diff > 1e-6:  # a tiny infraction of the bound
        LOGGER.debug("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
