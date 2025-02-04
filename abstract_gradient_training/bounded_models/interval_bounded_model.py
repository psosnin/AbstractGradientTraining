"""Pytorch model bounded with interval bounds."""

from __future__ import annotations

import logging
from typing import Literal
from collections.abc import Callable

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.bounded_models import nominal_modules

LOGGER = logging.getLogger(__name__)


class IntervalBoundedModel(BoundedModel):
    """
    A torch.nn.Sequential model bounded via interval bound propagation. For interface details, see the abstract base
    class `BoundedModel`. Below are some important notes on the implementation and usage of this class.

    Supported PyTorch modules:

        - torch.nn.Linear
        - torch.nn.ReLU
        - torch.nn.Conv2d
        - torch.nn.Flatten
        - torch.nn.Dropout

    PyTorch `requires_grad` attribute:

        This module does not respect the `requires_grad` attribute of the parameters of the original model, they will
        all be trained. If some parameters are to be fixed (i.e. fine-tuning), then the setting below should be used.

    Fine-tuning:

        Fine-tuning is supported only for the following setting: the original model is first partitioned into two parts:
            input -> fixed layers -> trainable layers -> output.
        The fixed layers may be passed in as a separate BoundedModel instance via the `transform` argument.

    Dropout layers:

        Dropout layers are supported, but they must be used with caution. The activation pattern for the dropout layer
        are sampled during the forward pass, and the same pattern is used for the bounded forward pass. This means that
        the a forward pass must be called first, and the same dropout mask will be used for all subsequent bounded
        forward passes until the next forward pass is called. Dropout is only applied when `retain_intermediate=True`,
        i.e. the forward pass is in training mode.

    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        *,
        transform: BoundedModel | None = None,
        trainable: bool = True,
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
    ):
        """
        Args:
            model (torch.nn.Sequential): Pytorch sequential model.
            transform (BoundedModel, optional): Fixed (non-trainable) transform which we pass the input through before
                the main model. This is useful for fine-tuning the last few layers of a pre-trained model.
            trainable (bool, optional): Flag to indicate if the model is trainable. If False, the model will not support
                gradient computation and all parameters will be fixed (non-interval).
            interval_matmul (str): one of ["rump", "exact", "nguyen"], method to use for interval matrix multiplication.
        """
        # check that the model only contains supported modules
        for module in model:
            if not isinstance(
                module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.Flatten, torch.nn.Dropout)
            ):
                raise ValueError(f"Unsupported module type in IBP: {type(module)}")
            if isinstance(module, torch.nn.Dropout):
                LOGGER.debug("Dropout layers must be used with caution, see IntervalBoundedModel for details.")
        if interval_matmul not in ["rump", "exact", "nguyen"]:
            raise ValueError(f"Unsupported method for interval matrix multiplication: {interval_matmul}")
        self.interval_matmul: Literal["rump", "exact", "nguyen"] = interval_matmul
        super().__init__(model, transform=transform, trainable=trainable)

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

        # loop over all the modules in the model and propagate the input interval
        for module, params_l, params_u in zip(self.modules, self._param_l, self._param_u):
            if isinstance(module, torch.nn.Linear):
                W_l, W_u = params_l[0], params_u[0]
                x_l, x_u = interval_arithmetic.propagate_matmul(
                    x_l, x_u, W_l.T, W_u.T, self.interval_matmul  # type: ignore
                )
                if len(params_l) == 2:  # handle the case where the bias is not None
                    b_l, b_u = params_l[1], params_u[1]
                    x_l, x_u = x_l + b_l, x_u + b_u
            elif isinstance(module, torch.nn.Conv2d):
                W_l, W_u = params_l[0], params_u[0]
                b_l = params_l[1] if len(params_l) == 2 else None  # handle the case where the bias is None
                b_u = params_u[1] if len(params_l) == 2 else None
                x_l, x_u = interval_arithmetic.propagate_conv2d(
                    x_l,
                    x_u,
                    W_l,
                    W_u,
                    b_l,
                    b_u,
                    stride=module.stride,
                    padding=module.padding,  # type: ignore
                    dilation=module.dilation,
                    groups=module.groups,
                )
            elif isinstance(module, (torch.nn.ReLU, torch.nn.Flatten)):
                x_l, x_u = module(x_l), module(x_u)
            elif isinstance(module, nominal_modules.DropoutWrapper):
                if retain_intermediate:  # only dropout in training mode
                    x_l, x_u = module.forward(x_l), module.forward(x_u)
            else:
                raise ValueError(f"Unsupported module type in IBP: {type(module)}")
            interval_arithmetic.validate_interval(x_l, x_u, msg=f"bounds at output of {module}")
            inter_l.append(x_l)
            inter_u.append(x_u)

        # cache the intermediate values if required
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

        # list to store bounds on the grads wrt the parameters
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
                # compute the gradient wrt the bias of the module
                if module.bias is not None:
                    grads_params_l.append(dl_dy_l)
                    grads_params_u.append(dl_dy_u)
                # compute the gradient wrt the weights of the module
                dl_dW_l, dl_dW_u = interval_arithmetic.propagate_matmul(
                    dl_dy_l.unsqueeze(-1),
                    dl_dy_u.unsqueeze(-1),
                    inter_l.unsqueeze(-2),
                    inter_u.unsqueeze(-2),
                    self.interval_matmul,  # type: ignore
                )
                grads_params_l.append(dl_dW_l)
                grads_params_u.append(dl_dW_u)
                # compute the gradients wrt the input to the module
                dl_dy_l, dl_dy_u = interval_arithmetic.propagate_matmul(
                    dl_dy_l, dl_dy_u, params_l[0], params_u[0], self.interval_matmul  # type: ignore
                )
            elif isinstance(module, torch.nn.Conv2d):
                # compute the gradients wrt the bias of the module
                if module.bias is not None:
                    grads_params_l.append(dl_dy_l.sum(dim=(2, 3)))
                    grads_params_u.append(dl_dy_u.sum(dim=(2, 3)))
                # compute the gradients wrt the weights of the module. Ideally we would simply use
                # torch.nn.functional.grad.conv2d_weight, but this function only gives the reduced gradients, not the
                # per-sample gradients.
                # Instead, we'll use the approach from https://github.com/owkin/grad-cnns/tree/master along with Rump's
                # algorithm (since the gradient is still a linear transformation).
                weight_grad_transform, input_grad_transform = _get_conv_gradient_transforms(module, inter_l)
                dl_dW_l, dl_dW_u = interval_arithmetic.propagate_linear_transform(
                    inter_l, inter_u, dl_dy_l, dl_dy_u, transform=weight_grad_transform
                )
                grads_params_l.append(dl_dW_l)
                grads_params_u.append(dl_dW_u)
                # compute the gradient wrt the input to the module. Again the gradient is a linear transformation, so
                # we can use Rump's algorithm.
                dl_dy_l, dl_dy_u = interval_arithmetic.propagate_linear_transform(
                    params_l[0], params_u[0], dl_dy_l, dl_dy_u, transform=input_grad_transform
                )
            elif isinstance(module, torch.nn.ReLU):
                # compute the gradient wrt the input to the module
                dl_dy_l, dl_dy_u = interval_arithmetic.propagate_elementwise(
                    dl_dy_l, dl_dy_u, (inter_l > 0).float(), (inter_u > 0).float()
                )
            elif isinstance(module, torch.nn.Flatten):
                # compute the gradient wrt the input to the module
                dl_dy_l = torch.reshape(dl_dy_l, inter_l.size())
                dl_dy_u = torch.reshape(dl_dy_u, inter_u.size())
            elif isinstance(module, nominal_modules.DropoutWrapper):
                dl_dy_l, dl_dy_u = module.forward(dl_dy_l), module.forward(dl_dy_u)
            else:
                raise ValueError(f"Unsupported module type in IBP: {type(module)}")
            interval_arithmetic.validate_interval(dl_dy_l, dl_dy_u, msg=f"bounds of grad at inputs to {module}")

        # delete the intermediate values from the previous forward pass
        self._inter_l = None
        self._inter_u = None

        # reverse and return the list of gradients wrt the parameters
        grads_params_l.reverse()
        grads_params_u.reverse()
        return grads_params_l, grads_params_u

    def __repr__(self) -> str:
        modules_repr = "\n\t\t".join(repr(mod) for mod in self.modules)
        repr_strs = [
            f"self.modules=[\n\t\t{modules_repr}\n\t],",
            f"{self.interval_matmul=},",
            f"{self.trainable=},",
        ]
        repr_strs = "\n\t".join(repr_strs)
        return f"IntervalBoundedModel(\n\t{repr_strs}\n)"


def _get_conv_gradient_transforms(
    module: torch.nn.Conv1d | torch.nn.Conv2d, inter: torch.Tensor
) -> tuple[Callable, Callable]:
    """
    Helper function for getting the transforms for backpropagating bounds on the gradient of a conv layer,
    which is needed to pass into Rump's algorithm in interval_arithmetic.propagate_linear_transform.
    """

    def weight_gradient_transform(x_, dl_):
        return nominal_modules._conv_weight_gradient(module, x_, dl_)

    def input_gradient_transform(W_, dl_):
        return torch.nn.functional.grad.conv2d_input(  # type: ignore
            inter.shape,
            W_,
            dl_,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    return weight_gradient_transform, input_gradient_transform


if __name__ == "__main__":
    # test the nominal forward and backwards pass of a simple nn model
    test_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 4, 2, 1, dilation=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 4, 1, 0, bias=False),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(2592, 100, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
    )

    # generate dummy data, pass it through the network and compute the gradients
    batch, targets = torch.randn(10, 3, 28, 28), torch.randn(10, 1)
    out = test_model(batch)
    loss = torch.nn.functional.mse_loss(out, targets)
    grads = torch.autograd.grad(loss, test_model.parameters())  # type: ignore

    # create the interval bounded model
    bounded_model = IntervalBoundedModel(test_model)

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
