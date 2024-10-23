"""Helper functions for working with torch.nn models in AGT."""

from collections.abc import Callable
import torch
from abstract_gradient_training import interval_arithmetic


def get_parameters(model: torch.nn.Sequential) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Get the parameters of the pytorch model, which must consist of alternating linear and ReLU layers only.

    Args:
        model (torch.nn.Sequential): Pytorch model to extract the parameters from.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]: Tuple of lists of [W1, b1, ..., Wn, bn] .
    """
    params = []
    for i, layer in enumerate(model):
        if i % 2 == 0:
            assert isinstance(layer, torch.nn.Linear), f"Expected Linear layer, got {layer}."
            params.append((layer.weight, layer.bias))
        else:
            assert isinstance(layer, torch.nn.ReLU), f"Expected ReLU layer, got {layer}."
    params = [item for sublist in params for item in sublist]  # flatten the list
    params = [t if len(t.shape) == 2 else t.unsqueeze(-1) for t in params]  # reshape biases to [n x 1] instead of [n]
    params = [t.detach().clone() for t in params]
    return params, [p.clone() for p in params], [p.clone() for p in params]


def propagate_conv_layers(
    x: torch.Tensor, model: torch.nn.Sequential, epsilon: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate an input batch through the convolutional layers of a model. Here we assume that the conv layers are all
    at the start of the network with ReLU activations after each one.

    Args:
        x (torch.Tensor): [batchsize x input_dim x 1] tensor of inputs to the network.
        model (torch.nn.Sequential): Pytorch model to extract the parameters from.
        epsilon (float): Epsilon value for the interval propagation.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of the lower bounds and upper bounds of the output of the
                                           convolutional layers of the network.
    """
    # get the parameters of the conv layers
    conv_layers = [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]
    conv_parameters = []
    for l in conv_layers:
        bias = l.bias.detach() if l.bias is not None else l.bias
        conv_parameters.append((l.weight.detach(), bias, l.stride, l.padding, l.dilation))
    # propagate the input through the conv layers
    x_l, x_u = x - epsilon, x + epsilon
    for W, b, stride, padding, dilation in conv_parameters:
        x_l, x_u = interval_arithmetic.propagate_conv2d(
            x_l, x_u, W, W, b, b, stride=stride, padding=padding, dilation=dilation
        )
        x_l, x_u = torch.nn.functional.relu(x_l), torch.nn.functional.relu(x_u)
    x_l = x_l.flatten(start_dim=1)
    x_u = x_u.flatten(start_dim=1)
    return x_l.unsqueeze(-1), x_u.unsqueeze(-1)


def get_conv_model_transform(model: torch.nn.Sequential) -> Callable:
    """
    Given a sequential torch model consisting of alternating Conv2d and ReLU layers with a flatten at the end, return a
    function that will propagate an interval over the input through the fixed model.
    """
    for i, layer in enumerate(model):
        if i == len(model) - 1:
            assert isinstance(layer, (torch.nn.Flatten)), f"Expected Flatten layer, got {layer}."
        elif i % 2 == 0:
            assert isinstance(layer, torch.nn.Conv2d), f"Expected Conv2d layer, got {layer}."
        else:
            assert isinstance(layer, torch.nn.ReLU), f"Expected ReLU layer, got {layer}."

    def transform(x: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        return propagate_conv_layers(x, model, epsilon)

    return transform
