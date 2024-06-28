"""Tests for backward pass and loss bounding functions."""

from itertools import product
import sys
import os
from parameterized import parameterized
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from tests import utils
from abstract_gradient_training import nominal_pass
from abstract_gradient_training.bounds import interval_bound_propagation as ibp


@parameterized.expand(
    product(
        utils.SHAPES_REGRESSION,
        utils.EPSILONS,
        utils.BACKWARD_BOUNDS,
        utils.BATCHSIZES,
        utils.LOSS_FNS_REGRESSION,
        range(utils.N_SEEDS),
    )
)
def test_backward_bounds_regression(shape, epsilon, backward_bound, batchsize, loss_fn, seed):
    """
    Test the backward bounding functions for the regression case. We first check that
    the backward pass matches torch.autograd when epsilon = 0 then we check that the bounds
    are sound when epsilon != 0.
    """
    # generate the network parameters
    param_n, param_l, param_u = utils.generate_network(shape, seed)
    # generate the input batch and targets
    x = torch.randn(batchsize, shape[0], 1, requires_grad=True).double()
    input_bounds = (x - epsilon, x + epsilon)
    targets = 1 - 2 * torch.randn(size=(batchsize, 1, 1)).double()
    # perform the nominal pass
    activations_n = nominal_pass.nominal_forward_pass(x, param_n)
    loss = loss_fn[0](activations_n[-1], targets, reduction="none")
    _, _, dL = loss_fn[1](activations_n[-1], activations_n[-1], activations_n[-1], targets)
    grad_n = nominal_pass.nominal_backward_pass(dL, param_n, activations_n)
    # check that the nominal pass matches torch.autograd
    for j in range(activations_n[0].shape[0]):  # iterate over batch dimension
        for i, P in enumerate(param_n):
            g = torch.autograd.grad(loss[j], P, retain_graph=True)[0]
            assert torch.allclose(g, grad_n[i][j]), f"{i}, {g - grad_n[i][j]}"
    # perform the bounding pass with interval over the input
    activations_l, activations_u = ibp.bound_forward_pass(param_n, param_n, *input_bounds)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_n, param_n, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
    # perform the bounding pass with interval over the parameters
    activations_l, activations_u = ibp.bound_forward_pass(param_l, param_u, x, x)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_l, param_u, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
    # perform the bounding pass with interval over the input and parameters
    activations_l, activations_u = ibp.bound_forward_pass(param_l, param_u, *input_bounds)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_l, param_u, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)


@parameterized.expand(
    product(
        utils.SHAPES_BINARY_CLASSIFICAITON,
        utils.EPSILONS,
        utils.BACKWARD_BOUNDS,
        utils.BATCHSIZES,
        utils.LOSS_FNS_BINARY_CLASSIFICATION,
        range(utils.N_SEEDS),
    )
)
def test_backward_bounds_binary_classification(shape, epsilon, backward_bound, batchsize, loss_fn, seed):
    """
    Test the backward bounding functions for the binary classification case. We first check that
    the backward pass matches torch.autograd when epsilon = 0 then we check that the bounds
    are sound when epsilon != 0.
    """
    # generate the network parameters
    param_n, param_l, param_u = utils.generate_network(shape, seed)
    # generate the input batch and targets
    x = torch.randn(batchsize, shape[0], 1, requires_grad=True).double()
    input_bounds = (x - epsilon, x + epsilon)
    targets = torch.randint(0, shape[-1], size=(batchsize,))
    # perform the nominal pass
    activations_n = nominal_pass.nominal_forward_pass(x, param_n)
    logit_n = activations_n[-1]
    if loss_fn[0] is torch.nn.functional.hinge_embedding_loss:
        hinge_targets = 2 * targets - 1
        loss = loss_fn[0](logit_n.flatten(start_dim=1).double(), hinge_targets[:, None].double(), reduction="none")
    else:
        loss = loss_fn[0](logit_n.flatten(start_dim=1).double(), targets[:, None].double(), reduction="none")
    _, _, dL = loss_fn[1](logit_n, logit_n, logit_n, targets)
    grad_n = nominal_pass.nominal_backward_pass(dL, param_n, activations_n)
    # check that the nominal pass matches torch.autograd
    for j in range(activations_n[0].shape[0]):
        for i, P in enumerate(param_n):
            g = torch.autograd.grad(loss[j], P, retain_graph=True)[0]
            assert torch.allclose(g, grad_n[i][j]), f"{i}, {g - grad_n[i][j]}"
    # perform the bounding pass with interval over the input
    activations_l, activations_u = ibp.bound_forward_pass(param_n, param_n, *input_bounds)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_n, param_n, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
    # perform the bounding pass with interval over the parameters
    activations_l, activations_u = ibp.bound_forward_pass(param_l, param_u, x, x)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_l, param_u, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
    # perform the bounding pass with interval over the input and parameters
    activations_l, activations_u = ibp.bound_forward_pass(param_l, param_u, *input_bounds)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_l, param_u, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)


@parameterized.expand(
    product(
        utils.SHAPES_CLASSIFICATION,
        utils.EPSILONS,
        utils.BACKWARD_BOUNDS,
        utils.BATCHSIZES,
        utils.LOSS_FNS_CLASSIFICATION,
        range(utils.N_SEEDS),
    )
)
def test_backward_bounds_classification(shape, epsilon, backward_bound, batchsize, loss_fn, seed):
    """
    Test the backward bounding functions for the multi-classification case. We first check that
    the backward pass matches torch.autograd when epsilon = 0 then we check that the bounds
    are sound when epsilon != 0.
    """
    # generate the network parameters
    param_n, param_l, param_u = utils.generate_network(shape, seed)
    # generate the input batch and targets
    x = torch.randn(batchsize, shape[0], 1, requires_grad=True).double()
    input_bounds = (x - epsilon, x + epsilon)
    targets = torch.randint(0, shape[-1], size=(batchsize,))
    # perform the nominal pass
    activations_n = nominal_pass.nominal_forward_pass(x, param_n)
    logit_n = activations_n[-1]
    loss = loss_fn[0](logit_n.flatten(start_dim=1), targets, reduction="none")
    _, _, dL = loss_fn[1](logit_n, logit_n, logit_n, targets)
    grad_n = nominal_pass.nominal_backward_pass(dL, param_n, activations_n)
    # check that the nominal pass matches torch.autograd
    for j in range(activations_n[0].shape[0]):
        for i, P in enumerate(param_n):
            g = torch.autograd.grad(loss[j], P, retain_graph=True)[0]
            assert torch.allclose(g, grad_n[i][j]), f"{i}, {g - grad_n[i][j]}"
    # perform the bounding pass with interval over the input
    activations_l, activations_u = ibp.bound_forward_pass(param_n, param_n, *input_bounds)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_n, param_n, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
    # perform the bounding pass with interval over the parameters
    activations_l, activations_u = ibp.bound_forward_pass(param_l, param_u, x, x)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_l, param_u, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
    # perform the bounding pass with interval over the input and parameters
    activations_l, activations_u = ibp.bound_forward_pass(param_l, param_u, *input_bounds)
    dL_l, dL_u, _ = loss_fn[1](activations_l[-1], activations_u[-1], activations_n[-1], targets)
    grad_min, grad_max = backward_bound(dL_l, dL_u, param_l, param_u, activations_l, activations_u)
    utils.validate_sound(grad_min, grad_n)
    utils.validate_sound(grad_n, grad_max)
