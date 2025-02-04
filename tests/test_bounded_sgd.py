"""Tests for the BoundedSGD optimizer."""

from itertools import product
import pytest
import torch
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.bounded_losses import BoundedCrossEntropyLoss, BoundedMSELoss, BoundedBCEWithLogitsLoss
from abstract_gradient_training.bounded_optimizers import BoundedSGD


@pytest.mark.parametrize(
    "loss_name, in_dim, batchsize, lr, l2_reg, l1_reg",
    product(["mse", "ce", "bce"], [1, 10], [1, 10], [0.01, 1.0], [0.0, 0.1], [0.0, 0.1]),
)
def test_bounded_sgd(loss_name, in_dim, batchsize, lr, l2_reg, l1_reg):
    """
    Test the nominal forward and backwards pass of a simple nn model.
    """
    if l2_reg and l1_reg:
        pytest.skip("L1 and L2 regularisation not supported together, skipping test.")
    torch.manual_seed(0)
    out_dim = 5 if loss_name != "bce" else 1
    hidden_layers = 2
    hidden_dim = 10
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, out_dim))
    test_model = torch.nn.Sequential(*layers)
    bounded_test_model = IntervalBoundedModel(test_model)

    # apply standard pytorch sgd update
    batch = torch.randn(batchsize, in_dim)
    optimizer = torch.optim.SGD(test_model.parameters(), lr=lr, weight_decay=l2_reg)  # type: ignore
    out = test_model(batch)

    if loss_name == "mse":
        targets = torch.randn(size=(batchsize, out_dim))
        loss = torch.nn.functional.mse_loss(out, targets, reduction="sum") / batchsize
        bounded_loss = BoundedMSELoss(reduction="sum")
    elif loss_name == "ce":
        targets = torch.randint(0, out_dim, size=(batchsize,))
        loss = torch.nn.functional.cross_entropy(out, targets, reduction="mean")
        bounded_loss = BoundedCrossEntropyLoss(reduction="sum")
    elif loss_name == "bce":
        targets = torch.randint(0, out_dim, size=(batchsize, 1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, targets.float(), reduction="mean")
        bounded_loss = BoundedBCEWithLogitsLoss(reduction="sum")
    else:
        raise ValueError(f"Unknown loss name {loss_name}")

    loss += sum(l1_reg * p.norm(1) for p in test_model.parameters())  # l1_reg
    loss.backward()
    optimizer.step()

    # apply our bounded sgd update
    bounded_optimizer = BoundedSGD(bounded_test_model, learning_rate=lr, l2_reg=l2_reg, l1_reg=l1_reg)
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
    assert torch.allclose(out, out_n)

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
        assert torch.all(1e-7 + pn >= pl), f"{(pn - pl).min()}"
        assert torch.all(1e-7 + pu >= pn), f"{(pu - pn).min()}"

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
    print(grads_l, grads_u, grads_n)
    import copy

    paraml, paramn, paramu = (
        copy.deepcopy(bounded_test_model.param_l),
        copy.deepcopy(bounded_test_model.param_n),
        copy.deepcopy(bounded_test_model.param_u),
    )
    bounded_optimizer.step(grads_l, grads_n, grads_u)
    for i, (pl, pn, pu) in enumerate(
        zip(bounded_test_model.param_l, bounded_test_model.param_n, bounded_test_model.param_u)
    ):
        if (wrong_idx := ((pn - pl) < 0)).any():
            print(wrong_idx.nonzero())
        if (wrong_idx := ((pu - pn) < 0)).any():
            print(idx := wrong_idx.nonzero(as_tuple=True))
            print(paraml[i][idx], paramn[i][idx], paramu[i][idx])
            print(
                (1 - lr * l2_reg) * paraml[i][idx],
                (1 - lr * l2_reg) * paramn[i][idx],
                (1 - lr * l2_reg) * paramu[i][idx],
            )
            print(grads_l[i][idx], grads_n[i][idx], grads_u[i][idx])
            print(pl[idx], pn[idx], pu[idx])

        assert torch.all(1e-7 + pn >= pl), f"{(pn - pl).min()}"
        assert torch.all(1e-7 + pu >= pn), f"{(pu - pn).min()}"
