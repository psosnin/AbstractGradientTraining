"""
Pytorch model bounded with mixed-integer programming and its relaxations.
Suppported bound types:

    - MIQP: Mixed-integer quadratic program
    - MILP: Mixed-integer linear program relaxation
    - QCQP: Quadratically constrained quadratic program relaxation
    - LP: Linear program relaxation
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import torch
import gurobipy as gp
import numpy as np

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.bounded_models import _mip_bounds

LOGGER = logging.getLogger(__name__)


class MIPBoundedModel(IntervalBoundedModel):
    """
    A torch.nn.Sequential model bounded via mixed-integer programming or its relaxations. This class wraps a standard
    pytorch model and provides methods to formulate the forward pass as an optimization problem using gurobi. The
    backward pass bounds are computed simply using interval bound propagation. For interface details, see the parent
    class. The class supports the following torch.nn modules:

    - torch.nn.Linear
    - torch.nn.ReLU
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        *,
        transform: IntervalBoundedModel | None = None,
        trainable: bool = True,
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
        forward_bound_mode: Literal["miqp", "milp", "qcqp", "lp"] = "miqp",
        optimize_inter: bool = False,
        gurobi_kwargs: dict | None = None,
    ):
        """
        Args:
            model (torch.nn.Sequential): Pytorch sequential model.
            transform (BoundedModel, optional): Fixed (non-trainable) transform which we pass the input through before
                the main model. This is useful for fine-tuning the last few layers of a pre-trained model.
            trainable (bool, optional): Flag to indicate if the model is trainable. If False, the model will not support
                gradient computation and all parameters will be fixed (non-interval).
            interval_matmul (str, optional): Method to compute the interval matrix multiplication. Supported methods are
                "rump", "exact", and "nguyen".
            forward_bound_mode (str, optional): Method to compute the forward bounds. Supported methods are "miqp",
                "milp", "qcqp", and "lp".
            optimize_inter (bool, optional): Flag to indicate if intermediate bounds should be optimized. If False, the
                intermediate bounds are computed using interval bound propagation.
            gurobi_kwargs (dict, optional): Keyword arguments to pass to the gurobi model.
        """
        # check that the model only contains supported modules
        for module in model:
            if not isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
                raise ValueError(f"Unsupported module type in IBP: {type(module)}")
        super().__init__(model, transform=transform, trainable=trainable, interval_matmul=interval_matmul)
        if forward_bound_mode not in ["miqp", "milp", "qcqp", "lp"]:
            raise ValueError(f"Unsupported forward bound mode: {forward_bound_mode}")
        self.forward_bound_mode = forward_bound_mode
        self.optimize_inter = optimize_inter
        self.gurobi_kwargs = gurobi_kwargs or {}
        self.forward_pass_model = None

    def bound_forward(
        self, x_l: torch.Tensor, x_u: torch.Tensor, retain_intermediate: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bounded forward pass of the model.

        Args:
            x_l (torch.Tensor): Lower bound of inputs to the model.
            x_u (torch.Tensor): Upper bound of inputs to the model.
            retain_intermediate (bool, optional): Flag to indicate if intermediate values should be cached by the model.
                If True, the model stores the intermediate bounds of the model which are required for the backward pass.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the output of the model.
        """
        if retain_intermediate and not self.trainable:
            raise ValueError("Intermediate values should not be retained for non-trainable models.")
        interval_arithmetic.validate_interval(x_l, x_u, msg=f"model input bounds")
        # pass the input through the fixed transform if required
        if self.transform is not None:
            x_l, x_u = self.transform.bound_forward(x_l, x_u, retain_intermediate=False)
            interval_arithmetic.validate_interval(x_l, x_u, msg=f"bounds after transform")
        # list to store the intermediate bounds from the forward pass
        inter_l, inter_u = [x_l], [x_u]

        # process each batch independently
        batchsize = x_l.size(0)
        lower_bounds = []
        upper_bounds = []
        start = time.time()
        for i in range(batchsize):
            if i % (batchsize // 10 + 1) == 0:
                LOGGER.debug("Solved %s bounds for %d/%d instances.", self.forward_bound_mode, i, batchsize)
            xi_l = x_l[i]
            xi_u = x_u[i]
            act_l, act_u, model = self._bound_forward_pass_helper(
                xi_l,
                xi_u,
            )
            lower_bounds.append(act_l)
            upper_bounds.append(act_u)
            if i == 0:
                LOGGER.debug(_mip_bounds.get_gurobi_model_stats(model))

        avg_time = (time.time() - start) / batchsize
        LOGGER.debug(
            "Solved %s bounds for %d instances. Avg bound time %.2fs.", self.forward_bound_mode, batchsize, avg_time
        )

        # concatenate the results
        inter_l = [torch.stack([act[i] for act in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
        inter_u = [torch.stack([act[i] for act in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

        if retain_intermediate:
            self._inter_l = inter_l
            self._inter_u = inter_u

        return inter_l[-1], inter_u[-1]

    def _bound_forward_pass_helper(
        self,
        x_l: torch.Tensor,
        x_u: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], gp.Model]:
        """
        Compute bounds on a single input by solving a mixed-integer program using gurobi.

        Args:
            x_l (torch.Tensor): [input_dim] Lower bound on a single input to the network.
            x_u (torch.Tensor): [input_dim] Upper bound on a single input to the network.

        Returns:
            inter_l (list[np.ndarray]): list of lower bounds on the output of each module in the network.
            inter_u (list[np.ndarray]): list of upper bounds on the output of each module in the network.
        """
        # define model and set the model parameters
        # TODO: The best option would be to re-use the model between successive calls by modifying the RHS of the
        # gurobi constraints, instead of re-forming the model each time. That isn't trivial to implement though so
        # we'll leave it to future work.
        model = _mip_bounds.init_gurobi_model("Bounds")
        model.setParam("NonConvex", 2)
        for key, value in self.gurobi_kwargs.items():
            model.setParam(key, value)

        # set the model type
        if self.forward_bound_mode == "miqp":
            relax_bilinear = False
            relax_binaries = False
        elif self.forward_bound_mode == "milp":
            relax_bilinear = True
            relax_binaries = False
        elif self.forward_bound_mode == "qcqp":
            relax_bilinear = False
            relax_binaries = True
        elif self.forward_bound_mode == "lp":
            relax_bilinear = True
            relax_binaries = True
        else:
            raise ValueError(f"Unsupported forward bound mode: {self.forward_bound_mode}")

        # add the input variable
        device = x_l.device
        h = model.addMVar(x_l.cpu().numpy().shape, lb=x_l.cpu().numpy(), ub=x_u.cpu().numpy())
        inter_l = [x_l]
        inter_u = [x_u]

        # loop over each hidden layer
        for i, (module, params_l, params_u) in enumerate(zip(self.modules, self._param_l, self._param_u)):
            if isinstance(module, torch.nn.Linear):
                assert module.bias is not None, "Only linear layers with bias are supported."
                W_l, W_u = params_l[0].cpu().numpy(), params_u[0].cpu().numpy()
                b_l, b_u = params_l[1].cpu().numpy(), params_u[1].cpu().numpy()
                W = model.addMVar(W_l.shape, lb=W_l, ub=W_u)
                b = model.addMVar(b_l.shape, lb=b_l, ub=b_u)
                # add the bilinear term
                s = _mip_bounds.add_bilinear_matmul(model, W, h, W_l, W_u, x_l, x_u, relax_bilinear)  # s = W @ h
                # ibp bounds on the next variable
                x_l, x_u = interval_arithmetic.propagate_matmul(
                    x_l, x_u, params_l[0].T, params_u[0].T, self.interval_matmul  # type: ignore
                )
                x_l, x_u = x_l.squeeze() + params_l[1], x_u.squeeze() + params_u[1]
                # add the next variable as an mvar
                h = s + b
                # optimized bounds on the next variable
                if self.optimize_inter and i > 0:
                    x_l_optimized, x_u_optimized = _mip_bounds.bound_objective(model, h)
                    if np.isinf(x_l_optimized).any() or np.isinf(x_u_optimized).any():
                        LOGGER.debug(
                            "Inf in optimized bounds for layer %d, falling back to IBP. Consider increasing timeout.",
                            i,
                        )
                    x_l = torch.maximum(x_l, torch.from_numpy(x_l_optimized).to(device).to(x_l.dtype))
                    x_u = torch.minimum(x_u, torch.from_numpy(x_u_optimized).to(device).to(x_l.dtype))
                model.addConstr(h <= x_u.cpu().numpy())
                model.addConstr(h >= x_l.cpu().numpy())
            elif isinstance(module, torch.nn.ReLU):
                h, _ = _mip_bounds.add_relu_bigm(model, h, x_l, x_u, relax_binaries)  # type: ignore
                x_l, x_u = module(x_l), module(x_u)
            else:
                raise ValueError(f"Unsupported module type in IBP: {type(module)}")
            interval_arithmetic.validate_interval(x_l, x_u, msg=f"bounds at output of {module}")
            inter_l.append(x_l)
            inter_u.append(x_u)

        if not self.optimize_inter:
            x_l_optimized, x_u_optimized = _mip_bounds.bound_objective(model, h)
            inter_l[-1] = torch.maximum(inter_l[-1], torch.from_numpy(x_l_optimized).to(device).to(x_l.dtype))
            inter_u[-1] = torch.minimum(inter_u[-1], torch.from_numpy(x_u_optimized).to(device).to(x_l.dtype))

        return inter_l, inter_u, model

    def __repr__(self) -> str:
        modules_repr = "\n\t\t".join(repr(mod) for mod in self.modules)
        repr_strs = [
            f"self.modules=[\n\t\t{modules_repr}\n\t],",
            f"{self.trainable=},",
            f"{self.interval_matmul=},",
            f"{self.forward_bound_mode=},",
            f"{self.optimize_inter=},",
            f"{self.gurobi_kwargs=},",
        ]
        repr_strs = "\n\t".join(repr_strs)
        return f"MIPBoundedModel(\n\t{repr_strs}\n)"


if __name__ == "__main__":
    # test the nominal forward and backwards pass of a simple nn model
    test_model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1),
        torch.nn.ReLU(),
    )

    # generate dummy data, pass it through the network and compute the gradients
    batch, targets = 10 * torch.randn(2, 4), 10 * torch.randn(2, 1)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets)
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model
    for mode in ["miqp", "milp", "qcqp", "lp"]:
        bounded_model = MIPBoundedModel(test_model, forward_bound_mode=mode, optimize_inter=True)  # type: ignore

        # test the nominal forward and backward passes
        custom_out = bounded_model.forward(batch, retain_intermediate=True)
        loss_grad = 2 * (custom_out - targets)
        assert torch.allclose(out, custom_out)
        custom_grads = bounded_model.backward(loss_grad)
        custom_grads = [g.mean(dim=0) for g in custom_grads]  # reduce over the batch dimension
        for g1, g2 in zip(grads, custom_grads):
            assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()

        # test that the bounded passes give the same results with zero interval
        out_l, out_u = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
        assert torch.allclose(out, out_l), (out - out_l).abs().max()
        assert torch.allclose(out, out_u), (out - out_u).abs().max()
        custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad, loss_grad)
        custom_grads_l = [g.mean(dim=0) for g in custom_grads_l]  # reduce over the batch dimension
        custom_grads_u = [g.mean(dim=0) for g in custom_grads_u]  # reduce over the batch dimension
        for g1, g2, g3 in zip(grads, custom_grads_l, custom_grads_u):
            assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()
            assert torch.allclose(g1, g3, 1e-6, 1e-6), (g1 - g3).abs().max()
