"""
Pytorch model bounded with linear bound propagation. The bounds support backward mode linear bound propagation
and various relaxations of the ReLU activation function. The bounds are similar to those used in the CROWN,
FastLin and DeepPoly verification algorithms, but differ in that here we must support interval parameters.

We use the following formulation of linear bound propagation:

    - A Node is a unit in a computational graph representing a function with linear and concrete bounds.
    - Bounds can be computed by propagating the linear bounds to form a linear relaxation of the whole model.
    - Since the parameters of the model are intervals, the bound propagation is computed using interval arithmetic.
    - The final bounds are computed via interval arithmetic on the linear bounds of the entire model.
    - The model also supports relaxation optimization (i.e. Alpha-CROWN) for the ReLU activation function.

Example:

    A Linear -> ReLU -> Linear model is converted to the following computational graph:

        (InputNode) l0 <= x0 <= u0
        (LinearNode) l1 <= Omega1 @ x0 + theta1 <= x1 = Linear(x0) <= Lambda1 @ x0 + delta1 <= u1
        (ReLUNode) l2 <= Omega2 @ x1 + theta2 <= x2 = ReLU(x1) <= Lambda2 @ x1 + delta2 <= u2
        (LinearNode) l3 <= Omega3 @ x2 + theta3 <= x3 = Linear(x2) <= Lambda3 @ x2 + delta3 <= u3

    The bound propagation procedure substitutes the linear bounds of each node into the next / previous node in the
    graph, resulting in a linear relaxation of the entire model:

        (Linear Relaxation) Omega @ x0 + theta <= x3 = model(x0) <= Lambda @ x0 + delta

    The bounds on the output of the model are then computed using interval arithmetic on the linear bounds.

        (Output Bounds) min(Omega @ [l0, u0] + theta) <= x3 <= max(Lambda @ [l0, u0] + delta)
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import IntervalBoundedModel, BoundedModel
from abstract_gradient_training.bounded_models import _crown_bounds

LOGGER = logging.getLogger(__name__)


class CROWNBoundedModel(IntervalBoundedModel):
    """
    A torch.nn.Sequential model bounded via linear bound propagation. The class supports the following torch.nn modules:

    - torch.nn.Linear
    - torch.nn.ReLU

    The forward pass bounds are always computed using linear bound propagation. The backward pass bounds can be computed
    using either interval propagation (which we inherit from IntervalBoundedModel) or linear bound propagation. The
    default is to use interval propagation for the backward pass, since we don't usually observe any advantage in LBP on
    the backward pass.

    This module does not respect the `requires_grad` attribute of the parameters of the original model, they will all be
    trained. If some parameters are to be fixed (i.e. fine-tuning), then the setting below should be used:

    Fine-tuning is supported only for the following setting: the original model is first partitioned into two parts:
        input -> fixed layers -> trainable layers -> output.
    The fixed layers may be passed in as a separate BoundedModel instance via the `transform` argument.
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        *,
        transform: BoundedModel | None = None,
        trainable: bool = True,
        relu_relaxation: Literal["zero", "one", "parallel", "optimizable"] = "zero",
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
        gradient_bound_mode: Literal["linear", "interval"] = "interval",
        alpha_crown_iters: int = 10,
        alpha_crown_lr: float = 0.1,
        optimize_inter: bool = False,
    ):
        """
        Args:
            model (torch.nn.Sequential): Pytorch sequential model.
            transform (BoundedModel, optional): Fixed (non-trainable) transform which we pass the input through before
                the main model. This is useful for fine-tuning the last few layers of a pre-trained model.
            trainable (bool, optional): Flag to indicate if the model is trainable. If False, the model will not support
                gradient computation and all parameters will be fixed (non-interval).
            relu_relaxation (str): one of ["zero", "one", "parallel", "optimizable"], method to use for ReLU relaxation.
            interval_matmul (str): one of ["rump", "exact", "nguyen"], method to use for interval matrix multiplication.
            gradient_bound_mode (str): one of ["linear", "interval"], method to use for gradient bound computation.
            alpha_crown_iters (int): number of iterations to use for Alpha-CROWN optimization.
            alpha_crown_lr (float): learning rate to use for Alpha-CROWN optimization.
            optimize_inter (bool, optional): Flag to indicate whether to optimize intermediates with Alpha-CROWN.
        """
        super().__init__(model, transform=transform, trainable=trainable, interval_matmul=interval_matmul)
        for module in model:
            if not isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
                raise ValueError(f"Unsupported module type in LBP: {type(module)}")
        if relu_relaxation not in ["zero", "one", "parallel", "optimizable"]:
            raise ValueError(f"Unsupported method for ReLU relaxation: {relu_relaxation}")
        if alpha_crown_iters == 0 and relu_relaxation == "optimizable":
            raise ValueError("Optimizable ReLU relaxation requires setting alpha-CROWN iterations > 0")
        if alpha_crown_lr < 0:
            raise ValueError("Alpha-CROWN requires a non-negative learning rate")
        if gradient_bound_mode not in ["linear", "interval"]:
            raise ValueError(f"Unsupported method for gradient bound computation: {gradient_bound_mode}")

        self.relu_relaxation = relu_relaxation
        self.gradient_bound_mode = gradient_bound_mode
        self.alpha_crown_iters = alpha_crown_iters
        self.alpha_crown_lr = alpha_crown_lr
        self.optimize_inter = optimize_inter

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
        interval_arithmetic.validate_interval(x_l, x_u, msg="Input bounds")

        # initialise the input node
        x = _crown_bounds.InputNode(x_l, x_u)
        inter_l = [x_l]
        inter_u = [x_u]

        # form the CROWN graph
        for module, params_l, params_u in zip(self.modules, self._param_l, self._param_u):
            if isinstance(module, torch.nn.Linear):
                assert module.bias is not None, "Linear layer without bias is not supported"
                W_l, b_l = params_l
                W_u, b_u = params_u
                x = _crown_bounds.AffineNode(x, W_l, W_u, b_l, b_u, interval_matmul=self.interval_matmul)
            elif isinstance(module, torch.nn.ReLU):
                x = _crown_bounds.ReLUNode(x, relu_lb=self.relu_relaxation, interval_matmul=self.interval_matmul)  # type: ignore
            else:
                raise ValueError(f"Unsupported module type in LBP: {type(module)}")
            if self.relu_relaxation == "optimizable" and self.optimize_inter:
                x_l, x_u = self.optimize_bounds(x)
            x_l, x_u = x.concretize()
            inter_l.append(x_l)
            inter_u.append(x_u)

        if self.relu_relaxation == "optimizable":
            # perform alpha-CROWN optimization
            x_l, x_u = self.optimize_bounds(x)

        if retain_intermediate:
            self._inter_l = inter_l
            self._inter_u = inter_u

        return x_l, x_u

    def bound_backward(
        self, dl_dy_l: torch.Tensor, dl_dy_u: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Bounded backward pass of the model. Note that the gradient bounds are computed with respect to the last forward
        pass call with retain_intermediate=True.

        Args:
            dl_dy_l (torch.Tensor): Lower bound of the gradient of the loss with respect to the output of the model.
            dl_dy_u (torch.Tensor): Upper bound of the gradient of the loss with respect to the output of the model.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Lower and upper bounds on the gradients of the loss with
                respect to the parameters of the model. The lists of the gradient tensors should be in the same order as
                that returned by the param_n, param_l, and param_u properties.
        """
        if self.gradient_bound_mode == "interval":
            return super().bound_backward(dl_dy_l, dl_dy_u)  # superclass is IntervalBoundedModel

        if not self.trainable:
            raise ValueError("Model is not trainable, so backward computations are not supported.")
        if self._inter_l is None or self._inter_u is None:
            raise ValueError("Bounded forward pass with retain_intermediate=True must be called before backward pass.")
        interval_arithmetic.validate_interval(dl_dy_l, dl_dy_u, msg=f"input gradient bounds")

        # compare gradient argument with the forward pass output
        if dl_dy_l.shape != (out_l := self._inter_l.pop()).shape:
            raise ValueError(f"Gradient shape does not match the forward pass output: {out_l.shape} != {dl_dy_l.shape}")
        if dl_dy_u.shape != (out_u := self._inter_u.pop()).shape:
            raise ValueError(f"Gradient shape does not match the forward pass output: {out_u.shape} != {dl_dy_u.shape}")

        # construct the input node to the lbp computational graph
        dl_dy = _crown_bounds.InputNode(dl_dy_l, dl_dy_u)
        grads_params_l = []
        grads_params_u = []
        # loop over the modules in reverse order and back-propagate the gradients
        for module, params_l, params_u, inter_l, inter_u in zip(
            reversed(self.modules),
            reversed(self._param_l),
            reversed(self._param_u),
            reversed(self._inter_l),
            reversed(self._inter_u),
        ):
            if isinstance(module, torch.nn.Linear):
                dl_dy_l, dl_dy_u = dl_dy.concretize()
                # compute the gradient wrt the bias of the module
                grads_params_l.append(dl_dy_l)
                grads_params_u.append(dl_dy_u)
                # compute the gradient wrt the weights of the module
                dl_dW_l, dl_dW_u = interval_arithmetic.propagate_elementwise(
                    dl_dy_l.unsqueeze(-1), dl_dy_u.unsqueeze(-1), inter_l.unsqueeze(-2), inter_u.unsqueeze(-2)
                )
                grads_params_l.append(dl_dW_l)
                grads_params_u.append(dl_dW_u)
                # extend the lbp computational graph
                dl_dy = _crown_bounds.AffineNode(
                    dl_dy, params_l[0].T, params_u[0].T, interval_matmul=self.interval_matmul
                )
            elif isinstance(module, torch.nn.ReLU):
                # extend the lbp computational graph
                dl_dy = _crown_bounds.MulNode(
                    dl_dy, (inter_l > 0).float(), (inter_u > 0).float(), interval_matmul=self.interval_matmul
                )
            else:
                raise ValueError(f"Unsupported module type in LBP: {type(module)}")
            interval_arithmetic.validate_interval(dl_dy_l, dl_dy_u, msg=f"bounds of grad at inputs to {module}")

        # delete the intermediate values from the previous forward pass
        self._inter_l = None
        self._inter_u = None

        # reverse and return the list of gradients wrt the parameters
        grads_params_l.reverse()
        grads_params_u.reverse()
        return grads_params_l, grads_params_u

    def optimize_bounds(self, node: _crown_bounds.Node) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize the lower bound slopes of any ReLU nodes using Alpha-CROWN.
        """
        if not node.optimizable_parameters():
            return node.concretize().as_tuple()
        optimizer = torch.optim.Adam(node.optimizable_parameters(), lr=self.alpha_crown_lr)  # type: ignore
        l, u = node.concretize()
        for i in range(self.alpha_crown_iters):
            optimizer.zero_grad()
            node.clear_cached()
            l, u = node.concretize()
            assert (l <= u).all()
            loss = (u - l).sum()
            LOGGER.debug("Alpha-CROWN iteration %d, loss=%s", i, loss)
            loss.backward(retain_graph=True)
            optimizer.step()
        return l, u

    def __repr__(self) -> str:

        modules_repr = "\n\t\t".join(repr(mod) for mod in self.modules)
        repr_strs = [
            f"self.modules=[\n\t\t{modules_repr}\n\t],",
            f"{self.trainable=},",
            f"{self.interval_matmul=},",
            f"{self.relu_relaxation=},",
            f"{self.gradient_bound_mode=},",
            f"{self.alpha_crown_iters=},",
            f"{self.alpha_crown_lr=},",
            f"{self.optimize_inter=},",
        ]
        repr_strs = "\n\t".join(repr_strs)
        return f"CROWNBoundedModel(\n\t{repr_strs}\n)"


if __name__ == "__main__":
    # test the nominal forward and backwards pass of a simple nn model
    test_model = torch.nn.Sequential(
        torch.nn.Linear(20, 10),
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )

    # generate dummy data, pass it through the network and compute the gradients
    batch, targets = torch.randn(4, 20), torch.randn(4, 1)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets)
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model
    bounded_model = CROWNBoundedModel(test_model, gradient_bound_mode="linear")

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
    assert torch.allclose(out, out_l, 1e-6, 1e-6), (out - out_l).abs().max()
    assert torch.allclose(out, out_u, 1e-6, 1e-6), (out - out_u).abs().max()
    custom_grads_l, custom_grads_u = bounded_model.bound_backward(loss_grad, loss_grad)
    custom_grads_l = [g.mean(dim=0) for g in custom_grads_l]  # reduce over the batch dimension
    custom_grads_u = [g.mean(dim=0) for g in custom_grads_u]  # reduce over the batch dimension
    for g1, g2, g3 in zip(grads, custom_grads_l, custom_grads_u):
        assert torch.allclose(g1, g2, 1e-6, 1e-6), (g1 - g2).abs().max()
        assert torch.allclose(g1, g3, 1e-6, 1e-6), (g1 - g3).abs().max()

    # test the alpha-crown bounds
    bounded_model = CROWNBoundedModel(
        test_model, gradient_bound_mode="linear", relu_relaxation="optimizable", alpha_crown_iters=10
    )
    out_l, out_u = bounded_model.bound_forward(batch - 0.1, batch + 0.1, retain_intermediate=True)
    assert (out_l <= out_u).all()
    print("Alpha-CROWN Bounds: ", (out_u - out_l).norm())

    bounded_model = CROWNBoundedModel(test_model, gradient_bound_mode="linear")
    out_l, out_u = bounded_model.bound_forward(batch - 0.1, batch + 0.1, retain_intermediate=True)
    assert (out_l <= out_u).all()
    print("CROWN bounds: ", (out_u - out_l).norm())
