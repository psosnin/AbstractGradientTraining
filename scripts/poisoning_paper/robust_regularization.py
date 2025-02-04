"""
This file implements the robust regularisation term from the paper 'Robust Explanation Constraints for Neural Networks'.
Nothing about this is 'certified', but we can use it for robust (but un-certified) pre-training.
"""

import torch
from abstract_gradient_training import interval_arithmetic


def input_gradient_interval_regularizer(
    model: torch.nn.Sequential,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    epsilon: float,
    model_epsilon: float,
    return_grads: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """
    Compute an interval over the gradients of the loss with respect to the inputs to the network. Then compute the norm
    of this input gradient interval and return it to be used as a regularization term. This can be used for robust
    (but un-certified) pre-training.

    Args:
        model (torch.nn.Sequential): The neural network model.
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize].
        loss_fn (str): Only "binary_cross_entropy" supported for now.
        epsilon (float): The training time input perturbation budget.
        model_epsilon (float): The training time model perturbation budget.
        return_grads (bool): Whether to return the gradients directly, instead of the regularization term.

    Returns:
        float | list[torch.Tensor]: The regularization term or the gradient lower bounds.
    """
    # we only support binary cross entropy loss for now
    if loss_fn != "binary_cross_entropy":
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    if isinstance(model[-1], torch.nn.Sigmoid):  # remove a signoid layer at the output if present
        model = model[:-1]

    # propagate through the forward pass
    intermediate = [(batch - epsilon, batch + epsilon)]
    for module in model:
        intermediate.append(propagate_module_forward(module, *intermediate[-1], model_epsilon))
        interval_arithmetic.validate_interval(*intermediate[-1], msg=f"forward pass {module}")

    # propagate through the loss function
    logits_l, logits_u = intermediate[-1]
    logits_l, logits_u = logits_l.squeeze(1), logits_u.squeeze(1)
    dl_l, dl_u = torch.sigmoid(logits_l) - labels, torch.sigmoid(logits_u) - labels
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="loss gradient")

    # propagate through the backward pass
    grads = [dl_l]
    for i in range(len(model) - 1, -1, -1):
        dl_l, dl_u = propagate_module_backward(model[i], dl_l, dl_u, *intermediate[i], model_epsilon)
        interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {model[i]}")
        grads.append(dl_l)

    if return_grads:
        grads.reverse()
        return grads
    return torch.norm(dl_u - dl_l, p=2) / dl_l.nelement()


def parameter_gradient_interval_regularizer(
    model: torch.nn.Sequential,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    epsilon: float,
    model_epsilon: float,
    return_grads: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """
    Compute an interval over the gradients of the loss with respect to the parameters to the network. Then compute the
    sum of the norms of the parameter gradient intervals and return it to be used as a regularization term. This can be
    used for robust (but un-certified) pre-training.

    Args:
        model (torch.nn.Sequential): The neural network model.
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize].
        loss_fn (str): Only "binary_cross_entropy" supported for now.
        epsilon (float): The training time input perturbation budget.
        model_epsilon (float): The training time model perturbation budget.
        return_grads (bool): Whether to return the gradients directly, instead of the regularization term.

    Returns:
        float | list[torch.Tensor]: The regularization term or the gradient lower bounds.
    """
    # we only support binary cross entropy loss for now
    if loss_fn != "binary_cross_entropy":
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    if isinstance(model[-1], torch.nn.Sigmoid):  # remove a signoid layer at the output if present
        model = model[:-1]

    # propagate through the forward pass
    intermediate = [(batch - epsilon, batch + epsilon)]
    for module in model:
        intermediate.append(propagate_module_forward(module, *intermediate[-1], model_epsilon))
        interval_arithmetic.validate_interval(*intermediate[-1], msg=f"forward pass {module}")

    # propagate through the loss function
    logits_l, logits_u = intermediate[-1]
    logits_l, logits_u = logits_l.squeeze(1), logits_u.squeeze(1)
    dl_l, dl_u = torch.sigmoid(logits_l) - labels, torch.sigmoid(logits_u) - labels
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="loss gradient")

    # propagate through the backward pass
    grads_l, grads_u = [], []
    for i in range(len(model) - 1, -1, -1):
        # compute the gradients wrt the parameters
        gl, gu = compute_module_parameter_gradients(model[i], dl_l, dl_u, *intermediate[i])
        grads_l.extend(gl)
        grads_u.extend(gu)
        # backpropagate the gradient through the module
        dl_l, dl_u = propagate_module_backward(model[i], dl_l, dl_u, *intermediate[i], model_epsilon)
        interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {model[i]}")

    # return the gradients or the regularization term
    if return_grads:
        grads_l.reverse()
        return grads_l

    regularization = torch.tensor(0, device=batch.device, dtype=batch.dtype)
    for l, u in zip(grads_l, grads_u):
        regularization += torch.norm(u - l, p=2)
    regularization /= sum(g.nelement() for g in grads_l)
    return regularization


def propagate_module_forward(
    module: torch.nn.Module, x_l: torch.Tensor, x_u: torch.Tensor, model_epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate the input interval through the given nn module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        x_l (torch.Tensor): Lower bound on the input.
        x_u (torch.Tensor): Upper bound on the input.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the output of the module.
    """
    if isinstance(module, torch.nn.Linear):
        W, b = module.weight, module.bias
        b = torch.zeros(W.size(0), 1, device=x_l.device, dtype=x_l.dtype) if b is None else b.unsqueeze(-1)
        W_l, W_u = W - model_epsilon, W + model_epsilon
        b_l, b_u = b - model_epsilon, b + model_epsilon
        if x_l.dim() == 2:
            x_l, x_u = x_l.unsqueeze(-1), x_u.unsqueeze(-1)
        x_l, x_u = interval_arithmetic.propagate_affine(x_l, x_u, W_l, W_u, b_l, b_u)
    elif isinstance(module, torch.nn.ReLU):
        x_l, x_u = torch.relu(x_l), torch.relu(x_u)
    elif isinstance(module, torch.nn.Conv2d):
        W, b = module.weight, module.bias
        W_l, W_u = W - model_epsilon, W + model_epsilon
        if b is None:
            b_l = torch.zeros(W.size(0), device=x_l.device, dtype=x_l.dtype)
            b_u = torch.zeros(W.size(0), device=x_l.device, dtype=x_l.dtype)
        else:
            b_l, b_u = b - model_epsilon, b + model_epsilon
        dilation, stride, padding = module.dilation, module.stride, module.padding
        x_l, x_u = interval_arithmetic.propagate_conv2d(
            x_l, x_u, W_l, W_u, b_l, b_u, stride=stride, padding=padding, dilation=dilation
        )
    elif isinstance(module, torch.nn.Flatten):
        x_l, x_u = module(x_l), module(x_u)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(x_l, x_u, msg=f"module {module}")
    return x_l, x_u


def propagate_module_backward(
    module: torch.nn.Module,
    dl_l: torch.Tensor,
    dl_u: torch.Tensor,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    model_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Back-propagate the gradient interval through the given nn module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        dl_l (torch.Tensor): Lower bound on the gradient of the loss wrt the output of this module.
        dl_u (torch.Tensor): Upper bound on the gradient gradient of the loss wrt the output of this module.
        x_l (torch.Tensor): Lower bound on the input to the module.
        x_u (torch.Tensor): Upper bound on the input to the module.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bound on the gradient of the loss wrt the module input.
    """
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="dl input")
    if isinstance(module, torch.nn.Linear):
        W = module.weight
        W_l, W_u = W - model_epsilon, W + model_epsilon
        dl_l, dl_u = interval_arithmetic.propagate_matmul(dl_l, dl_u, W_l, W_u)
    elif isinstance(module, torch.nn.ReLU):
        x_l, x_u = x_l.reshape(dl_l.size()), x_u.reshape(dl_u.size())
        dl_l, dl_u = interval_arithmetic.propagate_elementwise(dl_l, dl_u, (x_l > 0).float(), (x_u > 0).float())
    elif isinstance(module, torch.nn.Conv2d):
        W, dilation, stride = module.weight, module.dilation, module.stride
        padding, groups = module.padding, module.groups
        W_l, W_u = W - model_epsilon, W + model_epsilon

        # function that computes the gradient of the convolution wrt the input
        def conv_grad(dl_, W_):
            return torch.nn.grad.conv2d_input(x_l.shape, W_, dl_, stride, padding, dilation, groups)  # type: ignore

        dl_l, dl_u = interval_arithmetic.propagate_linear_transform(dl_l, dl_u, W_l, W_u, transform=conv_grad)
    elif isinstance(module, torch.nn.Flatten):
        dl_l, dl_u = torch.reshape(dl_l, x_l.size()), torch.reshape(dl_u, x_u.size())
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"module {module}")
    return dl_l, dl_u


def compute_module_parameter_gradients(
    module: torch.nn.Module,
    dl_l: torch.Tensor,
    dl_u: torch.Tensor,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute an interval over the gradients of the loss with respect to the parameters of the given module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        dl_l (torch.Tensor): Lower bound on the gradient of the loss wrt the output of this module.
        dl_u (torch.Tensor): Upper bound on the gradient gradient of the loss wrt the output of this module.
        x_l (torch.Tensor): Lower bound on the input to the module.
        x_u (torch.Tensor): Upper bound on the input to the module.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: List of tuples containing the lower and upper bounds on the
            gradients of the loss with respect to the parameters of the module.
    """
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="dl input")
    # get the parameters of the module
    parameters = list(module.parameters())
    # lists to store the gradient bounds
    grads_l, grads_u = [], []
    # if the module has no parameters return None
    if not parameters:
        return grads_l, grads_u

    # otherwise, compute gradients for the supported modules
    if isinstance(module, torch.nn.Linear):
        # compute gradients wrt the weight matrix
        dl_dW_l, dl_dW_u = interval_arithmetic.propagate_matmul(dl_l.T, dl_u.T, x_l.squeeze(-1), x_u.squeeze(-1))

        # compute gradients wrt the bias vector
        dl_db_l, dl_db_u = torch.sum(dl_l, dim=0), torch.sum(dl_u, dim=0)

        # store the gradients
        grads_l.extend([dl_db_l, dl_dW_l])
        grads_u.extend([dl_db_u, dl_dW_u])
    elif isinstance(module, torch.nn.Conv2d):
        # define gradient transform function
        W, dilation, stride, padding = module.weight, module.dilation, module.stride, module.padding

        # define the function that computes the gradient of the convolution wrt the weight matrix
        # note that this function is linear and hence we can compute an interval over the
        # output using Rump's algorithm.
        def transform(x, dl):
            return torch.nn.functional.grad.conv2d_weight(  # type: ignore
                x, W.size(), dl, stride=stride, padding=padding, dilation=dilation
            )

        # compute gradients
        dl_db_l, dl_db_u = torch.sum(dl_l, dim=(0, 2, 3)), torch.sum(dl_u, dim=(0, 2, 3))
        dl_dW_l, dl_dW_u = interval_arithmetic.propagate_linear_transform(x_l, x_u, dl_l, dl_u, transform)

        # store the gradients
        grads_l.extend([dl_db_l, dl_dW_l])
        grads_u.extend([dl_db_u, dl_dW_u])
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    return grads_l, grads_u


if __name__ == "__main__":
    # test the gradient calculations vs torch autograd
    import itertools

    # define model, dataset and optimizer
    device = "cuda:0"
    torch.manual_seed(0)

    test_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 4, 2, 0),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 4, 1, 0),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(3200, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
        torch.nn.Sigmoid(),
    )
    dataloader = itertools.cycle([(torch.rand(50, 3, 28, 28), torch.randint(0, 2, (50, 1)))])

    criterion = torch.nn.BCELoss(reduction="sum")
    test_model = test_model.to(device)

    # get the input data and pass through the model and loss
    test_batch, test_labels = next(iter(dataloader))
    test_batch, test_labels = test_batch.to(device), test_labels.to(device)
    test_batch.requires_grad = True
    intermediates = [test_batch]
    for layer in list(test_model.modules())[1:]:
        intermediates.append(layer(intermediates[-1]))
        intermediates[-1].retain_grad()
    loss = criterion(intermediates[-1].squeeze().float(), test_labels.squeeze().float())
    loss.backward()
    autograds_intermediate = [inter.grad for inter in intermediates]
    autograds_parameters = [param.grad for param in test_model.parameters()]

    # ======================== test the input_gradient_interval_regularizer ========================

    # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
    custom_grads = input_gradient_interval_regularizer(
        test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
    )

    for j, (cg, ag) in enumerate(zip(custom_grads, autograds_intermediate)):
        assert ag is not None
        print(f"Layer {j}: matching gradients = {torch.allclose(cg, ag, 1e-6, 1e-6)} {(cg - ag).abs().max()}")

    # gradient regularization term:
    reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0)
    print("Input gradient interval regularization term (no perturbation):", reg)
    assert reg == 0
    reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
    print("Input gradient interval regularization term (input perturbation):", reg)
    reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001)
    print("Input gradient interval regularization term (model perturbation):", reg)

    # ======================== test the parameter_gradient_interval_regularizer ========================

    # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
    custom_grads = parameter_gradient_interval_regularizer(
        test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
    )

    for j, (cg, ag) in enumerate(zip(custom_grads, autograds_parameters)):
        assert ag is not None
        print(f"Layer {j}: matching gradients = {torch.allclose(cg, ag, 1e-6, 1e-6)} {(cg - ag).abs().max()}")

    # gradient regularization term:
    reg = parameter_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0)
    assert reg == 0.0
    print("Parameter gradient interval regularization term (no perturbation):", reg)
    reg = parameter_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
    print("Parameter gradient interval regularization term (input perturbation):", reg)
    reg = parameter_gradient_interval_regularizer(
        test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001
    )
    print("Parameter gradient interval regularization term (model perturbation):", reg)
