import unittest
import sys
import os
from itertools import product

from parameterized import parameterized
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa

from abstract_gradient_training import interval_bound_propagation as ibp
from abstract_gradient_training import crown
from abstract_gradient_training import crown_ibp
from abstract_gradient_training import nominal_pass
from abstract_gradient_training import loss_gradient_bounds


"""
Declare different configurations to test.
"""


SHAPES = [
    (784, 32, 16, 10),  # MNIST 2 layer
    (784, 32, 10),  # MNIST 1 layer
    (10, 20, 10),  # larger hidden layer
]
EPSILONS = [0.01, 1]
BATCHSIZES = [1, 10]  # just to check a unit batchsize doesn't break anything
FORWARD_BOUNDS = [
    ibp.bound_forward_pass,
    crown.bound_forward_pass,
    crown_ibp.bound_forward_pass,
]
BACKWARD_BOUNDS = [
    ibp.bound_backward_pass,
    crown.bound_backward_pass,
]
N_SEEDS = 3  # number of seeds to test
LOSS_FNS = [
    "cross_entropy",
    "max_margin",
]


"""
Test Cases
"""


class TestBounds(unittest.TestCase):
    """
    Test cases that check that
        1. Each bounding function matches the nominal forward and backward passes exactly when
           epsilon = 0 and there is no weight perturbation.
        2. Each bounding function produces sound bounds when epsilon > 0 and there are weight
           perturbations. This is not a complete test of correctness but just a rough sanity check.
    """

    def initialise_attributes(self, shape, epsilon, batchsize, loss, seed):
        """
        Runs before all tests.
        """
        torch.random.manual_seed(seed)
        self.epsilon = epsilon
        # generate random NN parameters between -1 and 1
        self.param_n = []
        self.param_l = []
        self.param_u = []
        for i in range(len(shape) - 1):
            # keep requires_grad = True so we can check our gradients vs autograd
            W = torch.randn(shape[i+1], shape[i], requires_grad=True).double()
            b = torch.randn(shape[i+1], 1, requires_grad=True).double()
            self.param_n.append(W)
            self.param_n.append(b)
            self.param_l.append(W - torch.rand(W.shape))
            self.param_l.append(b - torch.rand(b.shape))
            self.param_u.append(W + torch.rand(W.shape))
            self.param_u.append(b + torch.rand(b.shape))
        # generate random input batch between -1 and 1
        x0 = torch.randn(batchsize, shape[0], 1, requires_grad=True).double()
        self.input_bounds = (x0 - epsilon, x0 + epsilon)
        self.labels = torch.randint(0, shape[-1], size=(batchsize, ))
        # perform the nominal forward pass
        self.logit, self.inter = nominal_pass.nominal_forward_pass(x0, self.param_n)
        if loss == "cross_entropy":
            # compute softmax and loss
            yl = F.log_softmax(self.logit, dim=1)
            self.loss = F.nll_loss(yl.flatten(start_dim=1), self.labels, reduction='none')
            self.grad_bound = loss_gradient_bounds.bound_cross_entropy_derivative
        elif loss == "max_margin":
            # compute max-margin loss
            self.loss = F.multi_margin_loss(self.logit.flatten(start_dim=1), self.labels, reduction='none')
            self.grad_bound = loss_gradient_bounds.bound_max_margin_derivative
        else:
            raise ValueError(f"Loss function {loss} not recognised")
        _, _, dL = self.grad_bound(self.logit, self.logit, self.logit, self.labels)
        grads = nominal_pass.nominal_backward_pass(dL, self.param_n, self.inter)
        self.validate_backward_pass_eq(self.loss, grads, grads)

    @parameterized.expand(product(SHAPES, EPSILONS, BATCHSIZES))
    def test_forward_passes(self, shape, epsilon, batchsize):
        """
        Test that the forward passes of the different bounding methods match the true forward pass when epsilon = 0 and
        give sound bounds when epsilon != 0.
        """
        loss = "binary_cross_entropy" if shape[-1] == 1 else "cross_entropy"
        for seed in range(N_SEEDS):
            self.initialise_attributes(shape, epsilon, batchsize, loss, seed)
            for forward in FORWARD_BOUNDS:
                with self.subTest([forward.__name__, shape, epsilon, batchsize, seed]):
                    # perform the bounding forward pass with no weight interval and no input perturbation
                    logit_l, logit_u, inter_l, inter_u = forward(self.param_n, self.param_n, self.inter[0], self.inter[0])  # noqa
                    # check that the output matches the nominal forward pass
                    self.validate_forward_pass_eq(self.logit, self.inter, logit_l, logit_u, inter_l, inter_u)

                    # perform the bounding forward pass with a weight interval and input perturbation
                    logit_l, logit_u, inter_l, inter_u = forward(self.param_l, self.param_u, *self.input_bounds)
                    # check that the output provides a sound bound
                    self.validate_forward_pass_sound(self.logit, self.inter, logit_l, logit_u, inter_l, inter_u)

    @parameterized.expand(product(SHAPES, EPSILONS, BATCHSIZES, LOSS_FNS))
    def test_backward_passes(self, shape, epsilon, batchsize, loss):
        """
        Test that the backward passes of the different bounding methods match the true backward pass when epsilon = 0
        and give sound bounds when epsilon != 0.
        """
        for seed in range(N_SEEDS):
            self.initialise_attributes(shape, epsilon, batchsize, loss, seed)
            for backward in BACKWARD_BOUNDS:
                with self.subTest([backward.__name__, shape, epsilon, batchsize, loss, seed]):
                    logit_l, logit_u, inter_l, inter_u = ibp.bound_forward_pass(self.param_n, self.param_n, self.inter[0], self.inter[0])  # noqa
                    # bound softmax output
                    dL_l, dL_u, _ = self.grad_bound(logit_l, logit_u, logit_u, self.labels)
                    # perform the bounding backward pass
                    grad_min, grad_max = backward(dL_l, dL_u, self.param_n, self.param_n, inter_l, inter_u)
                    # check the backwards pass against torch.autograd
                    self.validate_backward_pass_eq(self.loss, grad_min, grad_max)

                    logit_l, logit_u, inter_l, inter_u = ibp.bound_forward_pass(self.param_l, self.param_u, *self.input_bounds)
                    # bound softmax output
                    dL_l, dL_u, _ = self.grad_bound(logit_l, logit_u, logit_u, self.labels)
                    # perform the bounding backward pass
                    grad_min, grad_max = backward(dL_l, dL_u, self.param_l, self.param_u, inter_l, inter_u)
                    # check the backwards pass against torch.autograd
                    self.validate_backward_pass_sound(self.loss, grad_min, grad_max)

    def validate_forward_pass_eq(self, xhat, x, logit_l, logit_u, inter_l, inter_u):
        """
        Validate that the forward pass from the bounding function matches the nominal forward pass exactly.
        """
        # check that the logits are equivalent
        assert logit_l.shape == xhat.shape
        assert torch.allclose(logit_l, logit_u)
        assert torch.allclose(logit_l, xhat)
        # check that the intermediate bounds are equivalent
        assert len(inter_l) == len(inter_u)
        assert len(x) == len(inter_l)

        for i in range(len(x)):
            assert inter_l[i].shape == x[i].shape, f"Shape {inter_l[i].shape} does not match {x[i].shape}"
            assert inter_l[i].shape == inter_u[i].shape
            assert torch.allclose(inter_l[i], inter_u[i])
            assert torch.allclose(inter_l[i], x[i]), f"Error in layer {i} of {(inter_l[i] - x[i])}"

    def validate_backward_pass_eq(self, loss, grad_min, grad_max):
        """
        Validate that the backward pass from the bounding function matches the nominal backward pass exactly.
        """
        # check that the min and max grads are the same
        assert len(grad_min) == len(grad_max)
        for gmin, gmax in zip(grad_min, grad_max):
            assert torch.allclose(gmin, gmax), f"{gmin - gmax}"
        # check the backwards pass against torch.autograd
        # we have to for loop over the batchsize since autograd doesn't support a batched target
        for j in range(self.inter[0].shape[0]):
            for i, P in enumerate(self.param_n):
                g = torch.autograd.grad(loss[j], P, retain_graph=True)[0]
                assert torch.allclose(g, grad_min[i][j]), f"{i}, {g - grad_min[i][j]}"

    def validate_forward_pass_sound(self, xhat, x, logit_l, logit_u, inter_l, inter_u):
        """
        Validate that the bounds on the forward pass are sound.
        """
        # check that the logit bounds are sound
        assert logit_l.shape == xhat.shape
        assert torch.all(logit_l <= xhat)
        assert torch.all(xhat <= logit_u)
        # check that the intermediate bounds are sound
        assert len(inter_l) == len(inter_u)
        assert len(x) == len(inter_l)
        for i in range(len(x)):
            assert inter_l[i].shape == x[i].shape, f"Shape {inter_l[i].shape} does not match {x[i].shape}"
            assert inter_l[i].shape == inter_u[i].shape
            assert torch.all(inter_l[i] <= x[i])
            assert torch.all(x[i] <= inter_u[i])

    def validate_backward_pass_sound(self, loss, grad_min, grad_max):
        """
        Validate that the bounds on the forward pass are sound.
        """
        assert len(grad_min) == len(grad_max)
        # check the backwards pass against torch.autograd
        # we have to for loop over the batchsize since autograd doesn't support a batched target
        for j in range(self.inter[0].shape[0]):
            for i, P in enumerate(self.param_n):
                g = torch.autograd.grad(loss[j], P, retain_graph=True)[0]
                assert torch.all(grad_min[i][j] <= g), f"{i}, {g - grad_min[i][j]}"
                assert torch.all(g <= grad_max[i][j]), f"{i}, {grad_max[i][j] - g}"
