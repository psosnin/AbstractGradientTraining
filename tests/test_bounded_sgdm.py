"""Tests for the BoundedSGDM (momentum) optimizer."""

from itertools import product
import pytest
import torch
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.bounded_losses import BoundedCrossEntropyLoss, BoundedMSELoss, BoundedBCEWithLogitsLoss
from abstract_gradient_training.bounded_optimizers import BoundedSGDM


@pytest.mark.parametrize(
    "loss_name, in_dim, batchsize, lr, l2_reg, momentum, dampening, nesterov",
    product(
        ["mse", "ce", "bce"],
        [1, 10],
        [1, 10],
        [0.01, 0.1],
        [0.0, 0.1],
        [0.0, 0.9],
        [0.0, 0.9],
        [False, True],
    ),
)
def test_bounded_sgd(loss_name, in_dim, batchsize, lr, l2_reg, momentum, dampening, nesterov):
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    if nesterov == True and (momentum == 0 or dampening > 0):
        return  # Nesterov momentum is not supported with dampening or zero momentum

    torch.manual_seed(0)
    out_dim = 5 if loss_name != "bce" else 1
    hidden_layers = 2
    hidden_dim = 10
    n_iters = 4  # testing for more iters reveals numerical differences, but we're not overly concerned with that
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, out_dim))
    test_model = torch.nn.Sequential(*layers)
    bounded_test_model = IntervalBoundedModel(test_model)

    # apply standard pytorch sgd update
    batch = torch.randn(batchsize, in_dim)
    optimizer = torch.optim.SGD(  # type: ignore
        test_model.parameters(), lr=lr, weight_decay=l2_reg, momentum=momentum, dampening=dampening, nesterov=nesterov
    )

    if loss_name == "mse":
        targets = torch.randn(size=(batchsize, out_dim))
        loss_fn = lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="sum") / batchsize
        bounded_loss = BoundedMSELoss(reduction="sum")
    elif loss_name == "ce":
        targets = torch.randint(0, out_dim, size=(batchsize,))
        loss_fn = lambda x, y: torch.nn.functional.cross_entropy(x, y, reduction="mean")
        bounded_loss = BoundedCrossEntropyLoss(reduction="sum")
    elif loss_name == "bce":
        targets = torch.randint(0, out_dim, size=(batchsize, 1))
        loss_fn = lambda x, y: torch.nn.functional.binary_cross_entropy_with_logits(x, y.float(), reduction="mean")
        bounded_loss = BoundedBCEWithLogitsLoss(reduction="sum")
    else:
        raise ValueError(f"Unknown loss name {loss_name}")

    for _ in range(n_iters):
        optimizer.zero_grad()
        out = test_model(batch)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

    # apply our bounded sgd update
    bounded_optimizer = BoundedSGDM(
        bounded_test_model,
        learning_rate=lr,
        l2_reg=l2_reg,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
    )

    for _ in range(n_iters):
        out_n = bounded_test_model.forward(batch, retain_intermediate=True)
        loss_grad_n = bounded_loss.backward(out_n, targets)
        grads_n = bounded_test_model.backward(loss_grad_n)
        out_l, out_u = bounded_test_model.bound_forward(batch, batch, retain_intermediate=True)
        loss_grad_l, loss_grad_u = bounded_loss.bound_backward(out_l, out_u, targets)
        grads_l, grads_u = bounded_test_model.bound_backward(loss_grad_l, loss_grad_u)
        grads_l = [g.mean(dim=0) for g in grads_l]
        grads_u = [g.mean(dim=0) for g in grads_u]
        grads_n = [g.mean(dim=0) for g in grads_n]
        bounded_optimizer.step(grads_l, grads_n, grads_u)

    # test that the resulting models are the same
    out = test_model(batch)
    out_n = bounded_test_model.forward(batch)
    assert torch.allclose(out, out_n, 1e-6, 1e-6), f"{(out - out_n).abs().max()}"

    # test that the resulting parameters have valid bounds when an input interval is provided
    out_n = bounded_test_model.forward(batch, retain_intermediate=True)
    loss_grad_n = bounded_loss.backward(out_n, targets)
    grads_n = bounded_test_model.backward(loss_grad_n)
    out_l, out_u = bounded_test_model.bound_forward(batch - 0.01, batch + 0.01, retain_intermediate=True)
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(out_l, out_u, targets)
    grads_l, grads_u = bounded_test_model.bound_backward(loss_grad_l, loss_grad_u)
    grads_l = [g.mean(dim=0) for g in grads_l]
    grads_u = [g.mean(dim=0) for g in grads_u]
    grads_n = [g.mean(dim=0) for g in grads_n]
    bounded_optimizer.step(grads_l, grads_n, grads_u)
    for pl, pn, pu in zip(bounded_test_model.param_l, bounded_test_model.param_n, bounded_test_model.param_u):
        assert torch.all(1e-6 + pn >= pl), f"{pl - pn}"
        assert torch.all(1e-6 + pu >= pn), f"{pn - pu}"

    # test that the resulting parameters have valid bounds when a parameter interval is provided
    for pl, pu in zip(bounded_test_model.param_l, bounded_test_model.param_u):
        pl.data -= 0.001
        pu.data += 0.001
    out_n = bounded_test_model.forward(batch, retain_intermediate=True)
    loss_grad_n = bounded_loss.backward(out_n, targets)
    grads_n = bounded_test_model.backward(loss_grad_n)
    out_l, out_u = bounded_test_model.bound_forward(batch, batch, retain_intermediate=True)
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(out_l, out_u, targets)
    grads_l, grads_u = bounded_test_model.bound_backward(loss_grad_l, loss_grad_u)
    grads_l = [g.mean(dim=0) for g in grads_l]
    grads_u = [g.mean(dim=0) for g in grads_u]
    grads_n = [g.mean(dim=0) for g in grads_n]
    bounded_optimizer.step(grads_l, grads_n, grads_u)
    for pl, pn, pu in zip(bounded_test_model.param_l, bounded_test_model.param_n, bounded_test_model.param_u):
        assert torch.all(1e-6 + pn >= pl), f"{pl - pn}"
        assert torch.all(1e-6 + pu >= pn), f"{pn - pu}"
