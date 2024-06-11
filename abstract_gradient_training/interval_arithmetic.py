import os
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
    Compute an interval bound on the matrix multiplication A @ B.
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
    assert A_l.dim() >= 2, "A must be at least a 2D tensor"
    assert B_l.dim() >= 2, "B must be at least a 2D tensor"

    matmul_mode = os.environ.get("INTERVAL_MATMUL", "rump")

    if matmul_mode == "exact":
        return propagate_matmul_exact(A_l, A_u, B_l, B_u)
    elif matmul_mode == "rump":
        return propagate_matmul_rump(A_l, A_u, B_l, B_u)
    elif matmul_mode == "nguyen":
        return propagate_matmul_nguyen(A_l, A_u, B_l, B_u)
    elif matmul_mode == "distributive":
        return propagate_matmul_distributive(A_l, A_u, B_l, B_u)
    else:
        raise ValueError(f"Unknown interval matmul mode: {matmul_mode}")


def propagate_matmul_rump(A_l, A_u, B_l, B_u):
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


def propagate_matmul_exact(A_l, A_u, B_l, B_u):
    """
    Compute exact interval bounds on the matrix multiplication A @ B using exact interval arithmetic.
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
    E_l, E_u = propagate_elementwise(A_l.unsqueeze(-1), A_u.unsqueeze(-1), B_l.unsqueeze(-3), B_u.unsqueeze(-3))
    bound_utils.validate_interval(E_l, E_u)
    return E_l.sum(-2), E_u.sum(-2)


def propagate_matmul_distributive(A_l, A_u, B_l, B_u):
    """
    Compute interval bounds on the matrix multiplication A @ B using distributive interval arithmetic.
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
    # convert intervals from [l, u] form to [midpoint, absolute value] form
    A_mid = (A_l + A_u) / 2
    A_abs = (A_u - A_l) / 2 + A_mid.abs()
    B_mid = (B_l + B_u) / 2
    B_abs = (B_u - B_l) / 2 + B_mid.abs()
    # then compute the elementwise product A.unsqueeze(-1) * B.unsqueeze(-3) using distributive interval multiplication
    H_mid = A_mid.unsqueeze(-1) * B_mid.unsqueeze(-3)
    H_abs = A_abs.unsqueeze(-1) * B_abs.unsqueeze(-3)
    # now sum over the last dimension using distributive interval addition
    H_mid = H_mid.sum(-2)
    H_abs = H_abs.sum(-2)
    # convert back to [l, u] form
    H_rad = H_abs - H_mid.abs()
    H_l = H_mid - H_rad
    H_u = H_mid + H_rad
    bound_utils.validate_interval(H_l, H_u)
    return H_l, H_u


def propagate_matmul_nguyen(A_l, A_u, B_l, B_u):
    """
    Computes the interval bounds on the matrix multiplication A @ B given bounds on A and B using
    a decomposition of A into two intervals A^0 and A^* such that
    - A^0 is a zero-centered interval
    - A^* is an interval that does not contain zero
    """
    bound_utils.validate_interval(A_l, A_u)
    bound_utils.validate_interval(B_l, B_u)
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
