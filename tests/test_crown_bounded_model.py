"""
Test that the forward and backward passes computed by the interval bounded model are equivalent to the normal passes
computed by the original pytorch models.
"""

from itertools import product
import pytest
import torch
from abstract_gradient_training.bounded_models import CROWNBoundedModel
from abstract_gradient_training.bounded_losses import BoundedCrossEntropyLoss, BoundedMSELoss, BoundedBCEWithLogitsLoss


@pytest.mark.parametrize(
    "hidden_layers, hidden_dim, in_dim, out_dim, batchsize, relu_relaxation, gradient_bound_mode",
    product([1, 2], [1, 10], [1, 10], [1, 10], [1, 10], ["one", "zero", "parallel"], ["interval", "linear"]),
)
def test_fully_connected_model(
    hidden_layers, hidden_dim, in_dim, out_dim, batchsize, relu_relaxation, gradient_bound_mode
):
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    torch.manual_seed(0)
    # construct the test model
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, out_dim))
    test_model = torch.nn.Sequential(*layers)

    # construct dummy data and pass it through the pytorch test model
    batch, targets = torch.randn(batchsize, in_dim), torch.randn(batchsize, out_dim)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets, reduction="mean")
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model and pass the same data through it
    bounded_model = CROWNBoundedModel(
        test_model, relu_relaxation=relu_relaxation, gradient_bound_mode=gradient_bound_mode
    )
    custom_out = bounded_model.forward(batch, retain_intermediate=True)
    bounded_loss = BoundedMSELoss(reduction="mean")
    loss_grad = bounded_loss.backward(custom_out, targets)
    assert torch.allclose(out, custom_out)
    custom_grads = bounded_model.backward(loss_grad)
    custom_grads = [g.sum(dim=0) for g in custom_grads]  # reduce over the batch dimension
    for g1, g2 in zip(grads, custom_grads):
        assert g1.shape == g2.shape, (g1.shape, g2.shape)
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()

    # test that the bounded passes with zero interval also give the same results
    out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    assert torch.allclose(out, out_l), (out - out_l).abs().max()
    assert torch.allclose(out, out_u), (out - out_u).abs().max()
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(out_l, out_u, targets)
    custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad_l, loss_grad_u)
    custom_grads_l = [g.sum(dim=0) for g in custom_grads_l]  # reduce over the batch dimension
    custom_grads_u = [g.sum(dim=0) for g in custom_grads_u]  # reduce over the batch dimension
    for g1, g2, g3 in zip(grads, custom_grads_l, custom_grads_u):
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()
        assert torch.allclose(g1, g3, 1e-6, 1e-6), (g1 - g3).abs().max()


@pytest.mark.parametrize(
    "hidden_dim, in_dim, out_dim, batchsize, relu_relaxation",
    product([1, 10], [1, 10], [1, 10], [1, 10], ["one", "zero", "parallel"]),
)
def test_with_parameter_bounds(hidden_dim, in_dim, out_dim, batchsize, relu_relaxation):
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    torch.manual_seed(0)
    # construct the test model
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(2):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, out_dim))
    test_model = torch.nn.Sequential(*layers)

    # construct dummy data and pass it through the pytorch test model
    batch, targets = torch.randn(batchsize, in_dim), torch.randn(batchsize, out_dim)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets, reduction="mean")
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    bounded_model = CROWNBoundedModel(test_model, relu_relaxation=relu_relaxation, gradient_bound_mode="linear")
    bounded_loss = BoundedMSELoss(reduction="mean")

    # test that if we perturb the parameters of the model, the bounds are still valid
    for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
        p_l.data.add_(-torch.rand_like(p_l) * 0.1)
        p_u.data.add_(torch.rand_like(p_u) * 0.1)

    out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    assert torch.all(out_l <= out_u)
    for i_l, i_u in zip(bounded_model._inter_l, bounded_model._inter_u):  # type: ignore
        assert (i_l <= i_u).all()
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(out_l, out_u, targets)
    custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad_l, loss_grad_u)
    for g_l, g, g_u in zip(custom_grads_l, grads, custom_grads_u):
        assert (g <= g_u.sum(dim=0)).all()
        assert (g_l.sum(dim=0) <= g).all()


def test_with_alpha_crown():
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    torch.manual_seed(0)
    # construct the test model
    layers = [torch.nn.Linear(5, 10), torch.nn.ReLU()]
    for _ in range(2):
        layers.append(torch.nn.Linear(10, 10))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(10, 3))
    test_model = torch.nn.Sequential(*layers)

    # construct dummy data and pass it through the pytorch test model
    batch = torch.randn(6, 5)
    out = test_model(batch)

    bounded_model = CROWNBoundedModel(
        test_model, relu_relaxation="optimizable", gradient_bound_mode="interval", alpha_crown_iters=10
    )

    # test that if we perturb the parameters of the model, the bounds are still valid
    for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
        p_l.data.add_(-torch.rand_like(p_l) * 0.1)
        p_u.data.add_(torch.rand_like(p_u) * 0.1)

    out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    assert torch.all(out_l <= out_u)
    assert torch.all(out <= out_u)
    assert torch.all(out >= out_l)


@pytest.mark.parametrize(
    "in_dim, out_dim, batchsize",
    product(
        [1, 10],
        [1, 10],
        [1, 10],
    ),
)
def test_linear_model(in_dim, out_dim, batchsize):
    """
    Test the nominal forward and backwards pass of a simple linear model.
    """
    torch.manual_seed(0)
    # construct the test model
    test_model = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim))

    # construct dummy data and pass it through the pytorch test model
    batch, targets = torch.randn(batchsize, in_dim), torch.randn(batchsize, out_dim)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets, reduction="mean")
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model and pass the same data through it
    bounded_model = CROWNBoundedModel(test_model)
    custom_out = bounded_model.forward(batch, retain_intermediate=True)
    bounded_loss = BoundedMSELoss(reduction="mean")
    loss_grad = bounded_loss.backward(custom_out, targets)
    assert torch.allclose(out, custom_out)
    custom_grads = bounded_model.backward(loss_grad)
    custom_grads = [g.sum(dim=0) for g in custom_grads]  # reduce over the batch dimension
    for g1, g2 in zip(grads, custom_grads):
        assert g1.shape == g2.shape, (g1.shape, g2.shape)
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()

    # test that the bounded passes with zero interval also give the same results
    out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    assert torch.allclose(out, out_l), (out - out_l).abs().max()
    assert torch.allclose(out, out_u), (out - out_u).abs().max()
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(out_l, out_u, targets)
    custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad_l, loss_grad_u)
    custom_grads_l = [g.sum(dim=0) for g in custom_grads_l]  # reduce over the batch dimension
    custom_grads_u = [g.sum(dim=0) for g in custom_grads_u]  # reduce over the batch dimension
    for g1, g2, g3 in zip(grads, custom_grads_l, custom_grads_u):
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()
        assert torch.allclose(g1, g3, 1e-6, 1e-6), (g1 - g3).abs().max()


@pytest.mark.parametrize(
    "hidden_layers, hidden_dim, in_dim, out_dim, batchsize, relu_relaxation",
    product(
        [1, 2],
        [1, 10],
        [1, 10],
        [3, 10],
        [1, 10],
        ["one", "zero", "parallel"],
    ),
)
def test_linear_model_with_cross_entropy(hidden_layers, hidden_dim, in_dim, out_dim, batchsize, relu_relaxation):
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    torch.manual_seed(0)
    # construct the test model
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, out_dim))
    test_model = torch.nn.Sequential(*layers)

    # construct dummy data and pass it through the pytorch test model
    batch, targets = torch.randn(batchsize, in_dim), torch.randint(0, out_dim, size=(batchsize,))
    out = test_model(batch)
    loss = torch.nn.functional.cross_entropy(out, targets, reduction="mean")
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model and pass the same data through it
    bounded_model = CROWNBoundedModel(test_model, relu_relaxation=relu_relaxation)
    bounded_loss = BoundedCrossEntropyLoss(reduction="mean")
    custom_out = bounded_model.forward(batch, retain_intermediate=True)
    assert torch.allclose(out, custom_out)
    loss_grad = bounded_loss.backward(custom_out, targets)
    custom_grads = bounded_model.backward(loss_grad)
    custom_grads = [g.sum(dim=0) for g in custom_grads]  # reduce over the batch dimension
    for g1, g2 in zip(grads, custom_grads):
        assert g1.shape == g2.shape, (g1.shape, g2.shape)
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()

    # test that the bounded passes with zero interval also give the same results
    out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    assert torch.allclose(out, out_l), (out - out_l).abs().max()
    assert torch.allclose(out, out_u), (out - out_u).abs().max()
    custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad, loss_grad)
    custom_grads_l = [g.sum(dim=0) for g in custom_grads_l]  # reduce over the batch dimension
    custom_grads_u = [g.sum(dim=0) for g in custom_grads_u]  # reduce over the batch dimension
    for g1, g2, g3 in zip(grads, custom_grads_l, custom_grads_u):
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()
        assert torch.allclose(g1, g3, 1e-6, 1e-6), (g1 - g3).abs().max()


@pytest.mark.parametrize(
    "hidden_layers, hidden_dim, in_dim, batchsize, relu_relaxation",
    product(
        [1, 2],
        [1, 10],
        [1, 10],
        [1, 10],
        ["one", "zero", "parallel"],
    ),
)
def test_linear_model_with_binary_cross_entropy(hidden_layers, hidden_dim, in_dim, batchsize, relu_relaxation):
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    torch.manual_seed(0)
    # construct the test model
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, 1))
    test_model = torch.nn.Sequential(*layers)

    # construct dummy data and pass it through the pytorch test model
    batch, targets = torch.randn(batchsize, in_dim), torch.randint(0, 2, size=(batchsize, 1))
    out = test_model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, targets.float(), reduction="mean")
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model and pass the same data through it
    bounded_model = CROWNBoundedModel(test_model, relu_relaxation=relu_relaxation)
    bounded_loss = BoundedBCEWithLogitsLoss(reduction="mean")
    custom_out = bounded_model.forward(batch, retain_intermediate=True)
    assert torch.allclose(out, custom_out)
    loss_grad = bounded_loss.backward(custom_out, targets)
    custom_grads = bounded_model.backward(loss_grad)
    custom_grads = [g.sum(dim=0) for g in custom_grads]  # reduce over the batch dimension
    for g1, g2 in zip(grads, custom_grads):
        assert g1.shape == g2.shape, (g1.shape, g2.shape)
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()

    # test that the bounded passes with zero interval also give the same results
    out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
    assert torch.allclose(out, out_l), (out - out_l).abs().max()
    assert torch.allclose(out, out_u), (out - out_u).abs().max()
    custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad, loss_grad)
    custom_grads_l = [g.sum(dim=0) for g in custom_grads_l]  # reduce over the batch dimension
    custom_grads_u = [g.sum(dim=0) for g in custom_grads_u]  # reduce over the batch dimension
    for g1, g2, g3 in zip(grads, custom_grads_l, custom_grads_u):
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()
        assert torch.allclose(g1, g3, 1e-6, 1e-6), (g1 - g3).abs().max()
