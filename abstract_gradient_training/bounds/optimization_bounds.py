"""
Formulate the bounding problem as an optimization problem and solve it using Gurobi.
"""

import time
import logging
from typing import Any

import gurobipy as gp
import numpy as np
import torch

from abstract_gradient_training.bounds import bound_utils
from abstract_gradient_training.bounds import gurobi_utils
from abstract_gradient_training.bounds import mip_formulations
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounds import loss_gradients

# pyright: reportAssignmentType=false, reportArgumentType=false

LOGGER = logging.getLogger(__name__)


def bound_forward_pass(
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    x0_l: torch.Tensor,
    x0_u: torch.Tensor,
    *,
    relax_binaries: bool = False,
    relax_bilinear: bool = False,
    optimize_intermediate_bounds: bool = True,
    gurobi_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using the following formulations:

        - MIQP: The exact formulation
        - QCQP: Relax binary variables to continuous
        - MILP: Relax bilinear constraints to linear envelopes
        - LP: Relax both binary and bilinear constraints

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        optimize_intermediate_bounds (bool): Whether to solve an optimization problem for each intermediate activation
            or to use IBP bounds.
        gurobi_kwargs (dict | None): Parameters to pass to the gurobi model.

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on all (pre-relu) activations [x0, ..., xL] including
            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on all (pre-relu) activations [x0, ..., xL] including
            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
    """
    # validate the input
    param_l, param_u, x0_l, x0_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
    device = x0_l.device

    # get the name of the bounding method
    if relax_binaries and relax_bilinear:
        method = "LP"
    elif relax_binaries:
        method = "QCQP"
    elif relax_bilinear:
        method = "MILP"
    else:
        method = "MIQP"

    # convert all inputs to numpy arrays
    param_l = [param.detach().cpu().numpy() for param in param_l]
    param_u = [param.detach().cpu().numpy() for param in param_u]
    x0_l = x0_l.detach().cpu().numpy()
    x0_u = x0_u.detach().cpu().numpy()

    # iterate over each instance in the batch
    batchsize = x0_l.shape[0]
    lower_bounds = []
    upper_bounds = []
    start = time.time()
    for i in range(batchsize):
        if i % (batchsize // 10 + 1) == 0:
            LOGGER.debug("Solved %s bounds for %d/%d instances.", method, i, batchsize)
        x_l = x0_l[i]
        x_u = x0_u[i]
        act_l, act_u, model = _bound_forward_pass_helper(
            param_l, param_u, x_l, x_u, relax_binaries, relax_bilinear, optimize_intermediate_bounds, gurobi_kwargs
        )
        lower_bounds.append(act_l)
        upper_bounds.append(act_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved %s bounds for %d instances. Avg bound time %.2fs.", method, batchsize, avg_time)
    LOGGER.debug(gurobi_utils.get_gurobi_model_stats(model))

    # concatenate the results
    activations_l = [np.stack([act[i] for act in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
    activations_u = [np.stack([act[i] for act in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

    # convert the results back to torch tensors
    activations_l = [torch.tensor(act, device=device) for act in activations_l]
    activations_u = [torch.tensor(act, device=device) for act in activations_u]

    return activations_l, activations_u


def _bound_forward_pass_helper(
    param_l: list[np.ndarray],
    param_u: list[np.ndarray],
    x0_l: np.ndarray,
    x0_u: np.ndarray,
    relax_binaries: bool,
    relax_bilinear: bool,
    optimize_intermediate_bounds: bool,
    gurobi_kwargs: dict | None,
) -> tuple[list[np.ndarray], list[np.ndarray], gp.Model]:
    """
    Compute bounds on a single input by solving a mixed-integer program using gurobi.

    Args:
        param_l (list[np.ndarray]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[np.ndarray]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (np.ndarray): [input_dim x 1] Lower bound on a single input to the network.
        x0_u (np.ndarray): [input_dim x 1] Upper bound on a single input to the network.
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        optimize_intermedaite_bounds (bool): Whether to solve an optimization problem for each intermediate activation
            or to use IBP bounds.
        gurobi_kwargs (dict): Parameters to pass to the gurobi model.

    Returns:
        activations_l (list[np.ndarray]): list of lower bounds computed using bilinear programming on the (pre-relu)
            activations [x0, ..., xL] including the input and the logits.
        activations_u (list[np.ndarray]): list of upper bounds computed using bilinear programming on the (pre-relu)
            activations [x0, ..., xL] including the input and the logits.
        model (gp.Model): The gurobi model used to compute the bounds.
    """
    # define model and set the model parameters
    model = gurobi_utils.init_gurobi_model("Bounds")
    model.setParam("NonConvex", 2)
    if gurobi_kwargs:
        for key, value in gurobi_kwargs.items():
            model.setParam(key, value)

    # add the input variable
    h = model.addMVar(x0_l.shape, lb=x0_l, ub=x0_u)
    n_layers = len(param_l) // 2

    activations_l = [x0_l]
    activations_u = [x0_u]

    # loop over each hidden layer
    for i in range(0, n_layers):
        # define MVar for the weights and biases
        W_l, W_u = param_l[2 * i], param_u[2 * i]
        b_l, b_u = param_l[2 * i + 1], param_u[2 * i + 1]
        W = model.addMVar(W_l.shape, lb=W_l, ub=W_u)
        b = model.addMVar(b_l.shape, lb=b_l, ub=b_u)

        # bounds on the input to the current layer:
        h_l, h_u = activations_l[-1], activations_u[-1]
        if i > 0:
            h_l, h_u = np.maximum(h_l, 0), np.maximum(h_u, 0)

        # add the bilinear term s = W @ h
        s = mip_formulations.add_bilinear_matmul(model, W, h, W_l, W_u, h_l, h_u, relax_bilinear)

        # first compute the pre-activation bounds for this layer using IBP
        h_l, h_u = bound_utils.numpy_to_torch_wrapper(interval_arithmetic.propagate_matmul_exact, W_l, W_u, h_l, h_u)
        h_l, h_u = h_l + b_l, h_u + b_u

        # if i == 0, the best we can do is ibp.
        # if i == n_layers - 1, we are at the last layer and optimize the logit bounds
        # otherwise, only optimize the bounds for intermediate activations if the flag is set.
        if (i > 0 and optimize_intermediate_bounds) or i == n_layers - 1:
            h_l_optimized, h_u_optimized = gurobi_utils.bound_objective(model, s + b)
            if np.isinf(h_l_optimized).any() or np.isinf(h_u_optimized).any():
                LOGGER.debug(
                    "Inf in optimized bounds for layer %d, falling back to IBP. Consider increasing timeout.",
                    i,
                )
            h_l, h_u = np.maximum(h_l, h_l_optimized), np.minimum(h_u, h_u_optimized)

        # store the bounds
        activations_l.append(h_l)
        activations_u.append(h_u)

        # skip last layer
        if i == n_layers - 1:
            break

        # add next hidden variable
        h, _ = mip_formulations.add_relu_bigm(model, s + b, h_l, h_u, relax_binaries)

    return activations_l, activations_u, model


def bound_backward_pass(
    dL_min: torch.Tensor,
    dL_max: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    activations_l: list[torch.Tensor],
    activations_u: list[torch.Tensor],
    *,
    relax_binaries: bool = False,
    relax_bilinear: bool = False,
    relax_loss: bool = True,
    loss_fn: str = "cross_entropy",
    labels: torch.Tensor | None = None,
    gurobi_kwargs: dict | None = None,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters, intermediate activations and the first partial derivative of the loss, compute
    bounds on the gradients of the loss with respect to the parameters by solving an optimization problem.

    Args:
        dL_min (torch.Tensor): lower bound on the gradient of the loss with respect to the logits
        dL_max (torch.Tensor): upper bound on the gradient of the loss with respect to the logits
        param_l (list[torch.Tensor]): list of lower bounds on the parameters [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters [W1, b1, ..., Wm, bm]
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        relax_loss (bool): Whether to relax the loss function to interval propagation.
        loss_fn (str): The loss function to use if relax_loss is False.
        labels (torch.Tensor | None): Tensor of labels or targets for the batch, needed if relax_loss is False.
        gurobi_kwargs (dict | None): Parameters to pass to the gurobi model.

    Returns:
        grads_l (list[torch.Tensor]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[torch.Tensor]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
    """

    # validate the input
    dL_min, dL_max, param_l, param_u, activations_l, activations_u = bound_utils.validate_backward_bound_input(
        dL_min, dL_max, param_l, param_u, activations_l, activations_u
    )
    device = dL_min.device

    LOGGER.info("Optimization based bounds for backward pass not recommended, use IBP instead.")

    # convert all inputs to numpy arrays
    dL_min = dL_min.detach().cpu().numpy()
    dL_max = dL_max.detach().cpu().numpy()
    param_l = [param.detach().cpu().numpy() for param in param_l]
    param_u = [param.detach().cpu().numpy() for param in param_u]
    activations_l = [act.detach().cpu().numpy() for act in activations_l]
    activations_u = [act.detach().cpu().numpy() for act in activations_u]
    labels = labels.detach().cpu().numpy().squeeze() if labels is not None else None

    # get weight matrix bounds
    W_l, W_u = param_l[::2], param_u[::2]

    # iterate over each instance in the batch
    batchsize = activations_l[0].shape[0]
    lower_bounds = []
    upper_bounds = []
    start = time.time()
    for i in range(batchsize):
        if i % (batchsize // 10 + 1) == 0:
            LOGGER.debug("Solved backward pass bounds for %d/%d instances", i, batchsize)
        act_l = [act[i] for act in activations_l]
        act_u = [act[i] for act in activations_u]
        label = labels[i] if labels is not None else None
        d_l = dL_min[i]
        d_u = dL_max[i]
        g_l, g_u, model = _bound_backward_pass_helper(
            d_l, d_u, W_l, W_u, act_l, act_u, relax_binaries, relax_bilinear, relax_loss, loss_fn, label, gurobi_kwargs
        )

        lower_bounds.append(g_l)
        upper_bounds.append(g_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved backward pass bounds for %d instances. Avg bound time %.2fs.", batchsize, avg_time)
    LOGGER.debug(gurobi_utils.get_gurobi_model_stats(model))

    # concatenate the results
    grads_l = [np.stack([g[i] for g in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
    grads_u = [np.stack([g[i] for g in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

    # convert the results back to torch tensors
    grads_l = [torch.tensor(g, device=device) for g in grads_l]
    grads_u = [torch.tensor(g, device=device) for g in grads_u]

    return grads_l, grads_u


def _bound_backward_pass_helper(
    dL_min: np.ndarray,
    dL_max: np.ndarray,
    W_l: list[np.ndarray],
    W_u: list[np.ndarray],
    activations_l: list[np.ndarray],
    activations_u: list[np.ndarray],
    relax_binaries: bool,
    relax_bilinear: bool,
    relax_loss: bool,
    loss_fn: str,
    label: np.ndarray | None,
    gurobi_kwargs: dict | None,
) -> tuple[list[np.ndarray], list[np.ndarray], gp.Model]:
    """
    Compute backward pass bounds for a single input in the batch by formulating and solving an optimization problem
    for each gradient bound.

    Args:
        dL_min (np.ndarray): lower bound on the gradient of the loss with respect to the logits for a single input
        dL_max (np.ndarray): upper bound on the gradient of the loss with respect to the logits for a single input
        W_l (list[np.ndarray]): list of lower bounds on the weight matrices [W1, ..., Wm]
        W_u (list[np.ndarray]): list of upper bounds on the weight matrices [W1, ..., Wm]
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
            the input and the logits. Each tensor has shape [n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
            the input and the logits. Each tensor has shape [n_i x 1].
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        relax_loss (bool): Whether to relax the loss function to interval propagation.
        loss_fn (str): The loss function to use if relax_loss is False.
        label (np.ndarray | None): The label or target for the input, required if relax_loss is False.
        gurobi_kwargs (dict | None): Parameters to pass to the gurobi model.

    Returns:
        grads_l (list[np.ndarray]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[np.ndarray]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        model (gp.Model): The gurobi model used to compute the bounds.
    """

    # define model and set the model parameters
    model = gurobi_utils.init_gurobi_model("Bounds")
    model.setParam("NonConvex", 2)
    if gurobi_kwargs:
        for key, value in gurobi_kwargs.items():
            model.setParam(key, value)

    # add first partial derivative of the loss wrt the logits to the model
    if relax_loss:
        dL = model.addMVar(shape=dL_min.shape, lb=dL_min, ub=dL_max)
    else:
        assert label is not None, "Label is required if loss is not relaxed."
        act = model.addMVar(shape=activations_l[-1].shape, lb=activations_l[-1], ub=activations_u[-1])
        dL = mip_formulations.add_loss_gradient(model, act, label, loss_fn)

    # compute the gradient of the loss with respect to the weights and biases of the last layer
    dW_min, dW_max = bound_utils.numpy_to_torch_wrapper(
        interval_arithmetic.propagate_matmul,
        dL_min,
        dL_max,
        np.maximum(activations_l[-2].T, 0),
        np.maximum(activations_u[-2].T, 0),
        interval_matmul="exact",
    )
    grads_l, grads_u = [dL_min, dW_min], [dL_max, dW_max]

    # compute gradients for each layer
    for i in range(len(W_l) - 1, 0, -1):
        # initialise variable for the weight matrix
        W = model.addMVar(shape=W_l[i].shape, lb=W_l[i], ub=W_u[i])
        # add the bilinear term for s = W.T @ dL
        s = mip_formulations.add_bilinear_matmul(model, W.T, dL, W_l[i].T, W_u[i].T, dL_min, dL_max, relax_bilinear)
        # compute bounds on the next partial derivative
        dL_dz_min, dL_dz_max = gurobi_utils.bound_objective(model, s)
        # initialise variables for the next partial derivative, activation and heaviside of the activation
        dL_dz = model.addMVar(shape=(W_l[i].shape[1], 1), lb=dL_dz_min, ub=dL_dz_max)
        act = model.addMVar(shape=activations_l[i].shape, lb=activations_l[i], ub=activations_u[i])
        # note that we define the heaviside function using the pre-activation bounds
        heaviside = mip_formulations.add_heaviside(model, act, activations_l[i], activations_u[i], relax_binaries)
        # compute bounds on the heaviside term
        heaviside_l = np.heaviside(activations_l[i], 0)
        heaviside_u = np.heaviside(activations_u[i], 0)
        # add the bilinear term for dL_dz * heavi
        s = mip_formulations.add_bilinear_elementwise(
            model, dL_dz, heaviside, dL_dz_min, dL_dz_max, heaviside_l, heaviside_u, relax_bilinear
        )
        # compute bounds on the next partial derivvative
        dL_min, dL_max = gurobi_utils.bound_objective(model, s)
        # compute bounds on the partial derivative wrt the weights using ibp
        dW_min, dW_max = bound_utils.numpy_to_torch_wrapper(
            interval_arithmetic.propagate_matmul,
            dL_min,
            dL_max,
            np.maximum(activations_l[i - 1].T, 0) if i - 1 > 0 else activations_l[i - 1].T,
            np.maximum(activations_u[i - 1].T, 0) if i - 1 > 0 else activations_u[i - 1].T,
            interval_matmul="exact",
        )
        # store the results
        grads_l.append(dL_min)
        grads_l.append(dW_min)
        grads_u.append(dL_max)
        grads_u.append(dW_max)
        # add the next partial derivative variable
        dL = model.addMVar(shape=dL_min.shape, lb=dL_min, ub=dL_max)

    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u, model


def bound_forward_and_backward_pass(
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    x0_l: torch.Tensor,
    x0_u: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    *,
    relax_binaries: bool = False,
    relax_bilinear: bool = False,
    relax_loss: bool = True,
    optimize_intermediate_bounds: bool = True,
    gurobi_kwargs: dict | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the
    gradients of the loss with respect to the parameters by formulating and solving an optimization problem using
    gurobi.

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.
        labels (torch.Tensor): [batchsize x 1] Tensor of labels or targets for the batch.
        loss_fn (str): Name of the loss function to use.
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        relax_loss (bool): Whether to relax the loss function to interval propagation.
        optimize_intermediate_bounds (bool): Whether to solve an optimization problem for each intermediate activation
            or to use IBP bounds.
        gurobi_kwargs (dict | None): Parameters to pass to the gurobi model.

    Returns:
        grads_l (list[torch.Tensor]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[torch.Tensor]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
    """

    # validate the input
    param_l, param_u, x0_l, x0_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
    device = x0_l.device

    # convert all inputs to numpy arrays
    param_l = [param.detach().cpu().numpy() for param in param_l]
    param_u = [param.detach().cpu().numpy() for param in param_u]
    x0_l = x0_l.detach().cpu().numpy()
    x0_u = x0_u.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().squeeze()

    # iterate over each instance in the batch
    batchsize = x0_l.shape[0]
    lower_bounds = []
    upper_bounds = []
    start = time.time()
    for i in range(batchsize):
        if i % (batchsize // 10 + 1) == 0:
            LOGGER.debug("Solved combined bounds for %d/%d instances.", i, batchsize)
        x_l = x0_l[i]
        x_u = x0_u[i]
        label = labels[i] if labels is not None else None

        g_l, g_u, model = _bound_forward_and_backward_pass_helper(
            param_l,
            param_u,
            x_l,
            x_u,
            label,
            loss_fn,
            relax_binaries,
            relax_bilinear,
            relax_loss,
            optimize_intermediate_bounds,
            gurobi_kwargs,
        )

        lower_bounds.append(g_l)
        upper_bounds.append(g_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved combined bounds for %d instances. Avg bound time %.2fs.", batchsize, avg_time)
    LOGGER.debug(gurobi_utils.get_gurobi_model_stats(model))

    # concatenate the results
    grads_l = [np.stack([g[i] for g in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
    grads_u = [np.stack([g[i] for g in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

    # convert the results back to torch tensors
    grads_l = [torch.tensor(g, device=device) for g in grads_l]
    grads_u = [torch.tensor(g, device=device) for g in grads_u]

    return grads_l, grads_u


def _bound_forward_and_backward_pass_helper(
    param_l: list[np.ndarray],
    param_u: list[np.ndarray],
    x0_l: np.ndarray,
    x0_u: np.ndarray,
    label: np.ndarray,
    loss_fn: str,
    relax_binaries: bool,
    relax_bilinear: bool,
    relax_loss: bool,
    optimize_intermediate_bounds: bool,
    gurobi_kwargs: dict[str, Any] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], gp.Model]:
    """
    Given bounds on the parameters of the neural network and an interval over a single input point, compute bounds on
    the gradients of the loss with respect to the parameters by formulating and solving an optimization problem using
    gurobi.

    Args:
        param_l (list[np.ndarray]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[np.ndarray]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (np.ndarray): [input_dim x 1] Lower bound on a single input to the network.
        x0_u (np.ndarray): [input_dim x 1] Upper bound on a single input to the network.
        label (np.ndarray): The label or target for the input.
        loss_fn (str): Name of the loss function to use.
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        relax_loss (bool): Whether to relax the loss function to interval propagation.
        optimize_intermedaite_bounds (bool): Whether to solve an optimization problem for each intermediate activation
                                             or to use IBP bounds.
        gurobi_kwargs (dict): Parameters to pass to the gurobi model.

    Returns:
        grads_l (list[np.ndarray]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[np.ndarray]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        model (gp.Model): The gurobi model used to compute the bounds.
    """

    # define model and set the model parameters
    model = gurobi_utils.init_gurobi_model("Bounds")
    model.setParam("NonConvex", 2)
    if gurobi_kwargs:
        for key, value in gurobi_kwargs.items():
            model.setParam(key, value)

    # add the input variable
    h = model.addMVar(x0_l.shape, lb=x0_l, ub=x0_u)
    n_layers = len(param_l) // 2

    # list of activation bounds
    activations_l = [x0_l]
    activations_u = [x0_u]

    # list of activation, W and b variables
    W_vars = []
    act_vars = [h]
    post_act_vars = [h]
    act_heaviside_vars = [None]
    b_vars = []

    # loop over each hidden layer
    for i in range(0, n_layers):
        # define MVar for the weights and biases
        W_l, W_u = param_l[2 * i], param_u[2 * i]
        b_l, b_u = param_l[2 * i + 1], param_u[2 * i + 1]
        W_vars.append(model.addMVar(W_l.shape, lb=W_l, ub=W_u))
        b_vars.append(model.addMVar(b_l.shape, lb=b_l, ub=b_u))

        # bounds on the input to the current layer:
        h_l, h_u = activations_l[-1], activations_u[-1]
        if i > 0:
            h_l, h_u = np.maximum(h_l, 0), np.maximum(h_u, 0)

        # add the bilinear term s = W @ h
        s = mip_formulations.add_bilinear_matmul(model, W_vars[-1], h, W_l, W_u, h_l, h_u, relax_bilinear)

        # first compute the pre-activation bounds for this layer using IBP
        h_l, h_u = bound_utils.numpy_to_torch_wrapper(interval_arithmetic.propagate_matmul_exact, W_l, W_u, h_l, h_u)
        h_l, h_u = h_l + b_l, h_u + b_u

        # add the new activation variable to the model
        act_vars.append(s + b_vars[-1])

        # if i == 0, the best we can do is ibp.
        # if i == n_layers - 1, we are at the last layer and optimize the logit bounds
        # otherwise, only optimize the bounds for intermediate activations if the flag is set.
        if (i > 0 and optimize_intermediate_bounds) or i == n_layers - 1:
            h_l_optimized, h_u_optimized = gurobi_utils.bound_objective(model, act_vars[-1])
            if np.isinf(h_l_optimized).any() or np.isinf(h_u_optimized).any():
                LOGGER.debug(
                    "Inf in optimized bounds for layer %d, falling back to IBP. Consider increasing timeout.",
                    i,
                )
            h_l, h_u = np.maximum(h_l, h_l_optimized), np.minimum(h_u, h_u_optimized)

        # store the bounds
        activations_l.append(h_l)
        activations_u.append(h_u)

        # skip last layer
        if i == n_layers - 1:
            break

        # add next hidden variable
        h, z = mip_formulations.add_relu_bigm(model, act_vars[-1], h_l, h_u, relax_binaries, False)
        act_heaviside_vars.append(z)
        post_act_vars.append(h)

    # to re-use the pytorch and batched implementation, we have to do some reshaping and wrapping to
    # make the loss function derivative work
    dL_min, dL_max, _ = bound_utils.numpy_to_torch_wrapper(
        loss_gradients.bound_loss_function_derivative,
        loss_fn,
        activations_l[-1][None],
        activations_u[-1][None],
        activations_u[-1][None],
        np.array([label]),
    )
    dL_min, dL_max = dL_min.squeeze(0), dL_max.squeeze(0)
    # compute the gradient of the loss with respect to the weights and biases of the last layer
    dW_min, dW_max = bound_utils.numpy_to_torch_wrapper(
        interval_arithmetic.propagate_matmul,
        dL_min,
        dL_max,
        np.maximum(activations_l[-2].T, 0),
        np.maximum(activations_u[-2].T, 0),
        interval_matmul="exact",
    )

    # add first partial derivative of the loss wrt the logits to the model
    if relax_loss:
        dL = model.addMVar(shape=dL_min.shape, lb=dL_min, ub=dL_max)
    else:
        dL = mip_formulations.add_loss_gradient(model, act_vars[-1], label, loss_fn)

    grads_l, grads_u = [dL_min, dW_min], [dL_max, dW_max]

    W_l = param_l[::2]
    W_u = param_u[::2]

    # compute gradients for each layer
    for i in range(len(W_l) - 1, 0, -1):
        # add the bilinear term for s = W.T @ dL
        s = mip_formulations.add_bilinear_matmul(
            model, W_vars[i].T, dL, W_l[i].T, W_u[i].T, dL_min, dL_max, relax_bilinear
        )
        # compute bounds on the next partial derivative
        dL_dz_min, dL_dz_max = gurobi_utils.bound_objective(model, s)
        # initialise variables for the next partial derivative, activation and heaviside of the activation
        dL_dz = model.addMVar(shape=(W_l[i].shape[1], 1), lb=dL_dz_min, ub=dL_dz_max)
        # note that we define the heaviside function using the pre-activation bounds
        # compute bounds on the heaviside term
        heaviside_l = np.heaviside(activations_l[i], 0)
        heaviside_u = np.heaviside(activations_u[i], 0)
        # add the bilinear term for dL_dz * heavi
        s = mip_formulations.add_bilinear_elementwise(
            model, dL_dz, act_heaviside_vars[i], dL_dz_min, dL_dz_max, heaviside_l, heaviside_u, relax_bilinear
        )
        # compute bounds on the next partial derivvative
        dL_min, dL_max = gurobi_utils.bound_objective(model, s)
        # compute bounds on the partial derivative wrt the weights using ibp
        if relax_bilinear:
            z = mip_formulations.add_bilinear_elementwise(
                model,
                s,
                post_act_vars[i - 1].T,
                dL_min,
                dL_max,
                np.maximum(activations_l[i - 1].T, 0) if i - 1 > 0 else activations_l[i - 1].T,
                np.maximum(activations_u[i - 1].T, 0) if i - 1 > 0 else activations_u[i - 1].T,
                relax_bilinear,
            )
        else:
            t = model.addMVar(s.shape, lb=-np.inf)
            model.addConstr(t == s)
            z = t @ post_act_vars[i - 1].T
        dW_min, dW_max = gurobi_utils.bound_objective(model, z)
        # store the results
        grads_l.append(dL_min)
        grads_l.append(dW_min)
        grads_u.append(dL_max)
        grads_u.append(dW_max)
        # add the next partial derivative variable
        dL = model.addMVar(shape=dL_min.shape, lb=dL_min, ub=dL_max)

    # reverse the list of gradients
    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u, model
