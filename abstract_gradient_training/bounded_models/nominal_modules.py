"""
Nominal (non-interval) forward and backward passes of torch.nn.Modules. Backward passes must be implemented
for per-sample gradients, so we can't use torch.autograd.
"""

import torch
import numpy as np


def module_forward_pass(
    x: torch.Tensor, params: list[torch.Tensor], module: torch.nn.Module, train: bool = True
) -> torch.Tensor:
    """
    Forward pass through a torch.nn.Module with the specified parameters. The parameters to use for the transform are
    passed as a list of torch.Tensors and any parameter tensors within `module` itself are ignored. We still need the
    `module` object to determine the type of the module and access to additional attributes such as stride and padding.

    Args:
        x (torch.Tensor): Input to the module.
        params (list[torch.Tensor]): Parameters of the module.
        module (torch.nn.Module): Module used for information about the transformation, not for its parameters.
        train (bool): Whether to use the module in training mode. Default is True.

    Returns:
        torch.Tensor: Output of the module.
    """
    if isinstance(module, torch.nn.Linear):
        x = torch.nn.functional.linear(x, *params)  # this unpacking works even when bias=None
    elif isinstance(module, torch.nn.Conv2d):
        x = torch.nn.functional.conv2d(
            x,
            *params,
            stride=module.stride,
            padding=module.padding,  # type: ignore
            dilation=module.dilation,
            groups=module.groups,
        )
    elif isinstance(module, (torch.nn.ReLU, torch.nn.Flatten)):
        x = module(x)
    elif isinstance(module, DropoutWrapper):
        x = module(x, gen_mask=True) if train else x
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    return x


def module_backward_pass(
    dl_dy: torch.Tensor, x: torch.Tensor, params: list[torch.Tensor], module: torch.nn.Module
) -> list[torch.Tensor]:
    """
    Propagate the gradient through the module. Takes in the gradient of the loss wrt the output of the module, the input
    to the module and the parameters of the module. Returns a list of tensors, where the first tensor is the gradient
    of the loss wrt the input to the module, and the remaining tensors are the gradients of the loss wrt the parameters.
    The parameter gradients are returned per-sample, which is different from torch.autograd.grad.

    Args:
        dl_dy (torch.Tensor): Gradient of the loss wrt the output of the module.
        x (torch.Tensor): Input to the module.
        params (list[torch.Tensor]): Parameters of the module.
        module (torch.nn.Module): Module used for information about the transformation, not for its parameters.

    Returns:
        list[torch.Tensor]: Gradients of the loss wrt the input and parameters of the module.
    """

    grads = []

    if isinstance(module, torch.nn.Linear):
        # compute the gradients wrt the bias of the module
        if module.bias is not None:
            grads.append(dl_dy)
        # compute the gradients wrt the weights of the module (dl_db = dl_dy)
        dl_dW = dl_dy.unsqueeze(-1) * x.unsqueeze(1)
        grads.append(dl_dW)
        # compute the gradients wrt the input to the module
        dl_dy = dl_dy @ params[0]
    elif isinstance(module, torch.nn.Conv2d):
        # compute the gradients wrt the bias of the module
        if module.bias is not None:
            grads.append(dl_dy.sum(dim=(2, 3)))
        # compute the gradients wrt the weights of the module. Ideally we would simply use
        # torch.nn.functional.grad.conv2d_weight, but this function only gives the reduced gradients, not the
        # per-sample grads. Instead, we'll use the approach from https://github.com/owkin/grad-cnns/tree/master.
        dl_dW = _conv_weight_gradient(module, x, dl_dy)
        grads.append(dl_dW)
        # compute the gradient wrt the input to the module
        dl_dy = torch.nn.functional.grad.conv2d_input(  # type: ignore
            x.shape,
            params[0],
            dl_dy,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    elif isinstance(module, torch.nn.ReLU):
        # compute the gradient wrt the input to the module
        dl_dy = dl_dy * (x > 0).float()
    elif isinstance(module, torch.nn.Flatten):
        # compute the gradient wrt the input to the module
        dl_dy = torch.reshape(dl_dy, x.size())
    elif isinstance(module, DropoutWrapper):
        dl_dy = dl_dy * module.mask
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")

    return [dl_dy] + grads


def _conv_weight_gradient(
    module: torch.nn.Conv1d | torch.nn.Conv2d, module_input: torch.Tensor, module_output_grad: torch.Tensor
) -> torch.Tensor:
    """
    Computes per-sample gradients for nn.Conv1d and nn.Conv2d layers. This function is adapted from
    https://github.com/owkin/grad-cnns.

    The function computes per-sample gradients efficienly by shifting the batch dimension to the input channel dimension
    and using the 'groups' parameter of a convolutional layer of 1 dimension higher than the layer. For more detail on
    this procedure please see the linked repo and corresponding paper.

    Args:
        module (torch.nn.Conv1d | torch.nn.Conv2d): The convolutional layer for which to compute the gradients.
        module_input (torch.Tensor): The input to the convolutional layer.
        module_output_grad (torch.Tensor): The gradient of the loss with respect to the output of the layer.

    Returns:
        torch.Tensor: The per-sample gradients of the loss with respect to the weights of the convolutional layer.
    """
    if isinstance(module, torch.nn.Conv1d):
        nd = 1
        convnd = torch.nn.functional.conv2d
        s_ = np.s_[..., : module.kernel_size[0]]
    elif isinstance(module, torch.nn.Conv2d):
        nd = 2
        convnd = torch.nn.functional.conv3d
        s_ = np.s_[..., : module.kernel_size[0], : module.kernel_size[1]]
    else:
        raise NotImplementedError("Unsupported module type for convolutional per-sample gradients.")

    # change formats of kernel_size, stride, padding, and dilation to tuples
    stride = (module.stride,) * nd if isinstance(module.stride, int) else module.stride
    padding = (module.padding,) * nd if isinstance(module.padding, int) else module.padding
    dilation = (module.dilation,) * nd if isinstance(module.dilation, int) else module.dilation

    # Extract the batch size and input/output shapes
    batch_size = module_input.size(0)
    input_shape = module_input.size()[-nd:]
    output_shape = module_output_grad.size()[-nd:]

    # Reshape the input and gradient. Channels are seen as an extra spatial dimension with kernel size 1. The batch size
    # is seen as the channels using the 'groups' argument.
    module_input = module_input.view(1, batch_size * module.groups, module.in_channels // module.groups, *input_shape)
    module_output_grad = module_output_grad.view(-1, 1, 1, *output_shape)

    conv = convnd(
        module_input,
        module_output_grad,
        groups=batch_size * module.groups,
        stride=(1, *dilation),
        dilation=(1, *stride),
        padding=(0, *padding),  # type: ignore
    )

    # Reshape weight gradient and return
    return conv[s_].view(batch_size, *module.weight.shape).contiguous()


class DropoutWrapper(torch.nn.Dropout):
    """
    A wrapper around a torch.nn.Dropout module that stores the mask used during the forward pass. This is so the same
    dropout mask can be used for both the forward and bounded forward passes.
    """

    def __init__(self, dropout: torch.nn.Dropout):
        self.p = dropout.p
        self.mask = None
        assert not dropout.inplace, "Inplace dropout is not supported."
        super().__init__(p=dropout.p, inplace=dropout.inplace)

    def forward(self, input: torch.Tensor, gen_mask=False) -> torch.Tensor:
        if gen_mask:
            self.mask = input.bernoulli(1 - self.p)
        if self.mask is None:
            raise ValueError("No dropout mask found, you may be calling forward passes in the wrong order.")
        return input * self.mask

    def __call__(self, input: torch.Tensor, gen_mask=False) -> torch.Tensor:
        return self.forward(input, gen_mask)
