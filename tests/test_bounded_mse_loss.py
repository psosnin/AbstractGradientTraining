"""
Test that the forward and backward passes computed by the BoundedMSELoss are equivalent to the normal passes
computed by the original torch.nn.MSELoss. Also test that the bounds computed by the BoundedMSELoss are valid for
bounded inputs and for poisoning attacks.
"""

from itertools import product
import pytest
import torch
from abstract_gradient_training.bounded_losses import BoundedMSELoss


@pytest.mark.parametrize(
    "targets_dim, batchsize, reduction",
    product(
        [1, 10],
        [1, 10],
        ["sum", "mean", "none"],
    ),
)
def test_bounded_mse(
    targets_dim,
    batchsize,
    reduction,
):
    """
    Test the mean squared error bounded loss
    """
    torch.manual_seed(0)
    # construct the dummy data
    inputs, targets = torch.randn(batchsize, targets_dim, requires_grad=True), torch.randn(batchsize, targets_dim)
    loss = BoundedMSELoss(reduction=reduction)
    normal_loss = torch.nn.MSELoss(reduction=reduction)

    # test the unbounded forward and backward passes
    normal_loss_val = normal_loss(inputs, targets)
    loss_val = loss.forward(inputs, targets)
    assert torch.allclose(normal_loss_val, loss_val), f"Expected {normal_loss_val}, got {loss_val}"
    nominal_loss_grad = torch.autograd.grad(normal_loss_val.sum(), inputs, create_graph=True)[0]
    loss_grad = loss.backward(inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad), f"Expected {nominal_loss_grad}, got {loss_grad}"

    # test the bounded forward and backward passes
    loss_val_l, loss_val_u = loss.bound_forward(inputs, inputs, targets)
    assert torch.allclose(normal_loss_val, loss_val_l), f"Expected {normal_loss_val}, got {loss_val_l}"
    assert torch.allclose(normal_loss_val, loss_val_u), f"Expected {normal_loss_val}, got {loss_val_u}"
    loss_grad_l, loss_grad_u = loss.bound_backward(inputs, inputs, targets)
    assert torch.allclose(nominal_loss_grad, loss_grad_l), f"Expected {nominal_loss_grad}, got {loss_grad_l}"
    assert torch.allclose(nominal_loss_grad, loss_grad_u), f"Expected {nominal_loss_grad}, got {loss_grad_u}"

    # test that the bounds are valid for bounded inputs
    loss_val_l, loss_val_u = loss.bound_forward(inputs - 0.1, inputs + 0.1, targets)
    assert torch.all(loss_val_l <= loss_val_u), f"Expected lower bounds to be less than upper bounds"
    assert torch.all(loss_val_l <= loss_val), f"Expected lower bounds to be less than the loss"
    assert torch.all(loss_val <= loss_val_u), f"Expected loss to be less than upper bounds"
    loss_grad_l, loss_grad_u = loss.bound_backward(inputs - 0.1, inputs + 0.1, targets)
    assert torch.all(loss_grad_l <= loss_grad_u), f"Expected lower bounds to be less than upper bounds"
    assert torch.all(loss_grad_l <= loss_grad), f"Expected lower bounds to be less than the loss"
    assert torch.all(loss_grad <= loss_grad_u), f"Expected loss to be less than upper bounds"

    # test that the bounds are valid for poisoning attack
    loss_val_l, loss_val_u = loss.bound_forward(inputs, inputs, targets, label_k_poison=10, label_epsilon=0.1)
    assert torch.all(loss_val_l <= loss_val_u), f"Expected lower bounds to be less than upper bounds"
    assert torch.all(loss_val_l <= loss_val), f"Expected lower bounds to be less than the loss"
    assert torch.all(loss_val <= loss_val_u), f"Expected loss to be less than upper bounds"
    loss_grad_l, loss_grad_u = loss.bound_backward(inputs, inputs, targets, label_k_poison=10, label_epsilon=0.1)
    assert torch.all(loss_grad_l <= loss_grad_u), f"Expected lower bounds to be less than upper bounds"
    assert torch.all(loss_grad_l <= loss_grad), f"Expected lower bounds to be less than the loss"
    assert torch.all(loss_grad <= loss_grad_u), f"Expected loss to be less than upper bounds"
