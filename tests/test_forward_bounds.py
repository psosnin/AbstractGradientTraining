"""
Tests for forward bounding methods.
"""

from itertools import product
import sys
import os
from parameterized import parameterized
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from tests import utils
from abstract_gradient_training import nominal_pass


@parameterized.expand(
    product(
        utils.SHAPES_ALL,
        utils.FORWARD_BOUNDS,
        utils.EPSILONS,
        utils.BATCHSIZES,
        range(utils.N_SEEDS),
    )
)
def test_forward_bound(shape, forward_bound, epsilon, batchsize, seed):
    """
    Test the interval bound propagation method.
    """
    # generate the network parameters
    param_n, param_l, param_u = utils.generate_network(shape, seed)
    # generate the input batch
    x = torch.randn(batchsize, shape[0], 1, requires_grad=True).double()
    # do nominal pass
    activations_n = nominal_pass.nominal_forward_pass(x, param_n)
    # do bounding pass with no epsilon
    activations_l, activations_u = forward_bound(param_n, param_n, x, x)
    # validate the bounds
    utils.validate_equal(activations_l, activations_n)
    utils.validate_equal(activations_n, activations_u)
    # do bounding pass with bounds on parameters
    activations_l, activations_u = forward_bound(param_l, param_u, x, x)
    # validate the bounds
    utils.validate_sound(activations_l, activations_n)
    utils.validate_sound(activations_n, activations_u)
    # do bounding pass with bounds on input
    activations_l, activations_u = forward_bound(param_n, param_n, x - epsilon, x + epsilon)
    # validate the bounds
    utils.validate_sound(activations_l, activations_n)
    utils.validate_sound(activations_n, activations_u)
    # do bounding pass with bounds on input and parameters
    activations_l, activations_u = forward_bound(param_l, param_u, x - epsilon, x + epsilon)
    # validate the bounds
    utils.validate_sound(activations_l, activations_n)
    utils.validate_sound(activations_n, activations_u)
