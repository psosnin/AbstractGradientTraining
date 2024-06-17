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
    logit_n, inter_n = nominal_pass.nominal_forward_pass(x, param_n)
    # do bounding pass with no epsilon
    logit_l, logit_u, inter_l, inter_u = forward_bound(param_n, param_n, x, x)
    # validate the bounds
    utils.validate_equal(inter_l, inter_n)
    utils.validate_equal(inter_n, inter_u)
    utils.validate_equal(logit_l, logit_n)
    utils.validate_equal(logit_n, logit_u)
    # do bounding pass with bounds on parameters
    logit_l, logit_u, inter_l, inter_u = forward_bound(param_l, param_u, x, x)
    # validate the bounds
    utils.validate_sound(inter_l, inter_n)
    utils.validate_sound(inter_n, inter_u)
    utils.validate_sound(logit_l, logit_n)
    utils.validate_sound(logit_n, logit_u)
    # do bounding pass with bounds on input
    logit_l, logit_u, inter_l, inter_u = forward_bound(param_n, param_n, x - epsilon, x + epsilon)
    # validate the bounds
    utils.validate_sound(inter_l, inter_n)
    utils.validate_sound(inter_n, inter_u)
    utils.validate_sound(logit_l, logit_n)
    utils.validate_sound(logit_n, logit_u)
    # do bounding pass with bounds on input and parameters
    logit_l, logit_u, inter_l, inter_u = forward_bound(param_l, param_u, x - epsilon, x + epsilon)
    # validate the bounds
    utils.validate_sound(inter_l, inter_n)
    utils.validate_sound(inter_n, inter_u)
    utils.validate_sound(logit_l, logit_n)
    utils.validate_sound(logit_n, logit_u)
