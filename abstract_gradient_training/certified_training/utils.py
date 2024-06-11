import os
import logging
import torch
import copy

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training import definitions as agt_definitions

"""
Helper functions for certified training.
"""


def grads_helper(batch_l, batch_u, labels, param_l, param_u, config, label_poison=False):
    """
    Helper function to calculate bounds on the gradient of the loss function with respect to all parameters given the
    input and parameter bounds.
    Parameters:
        batch_l: [fragsize x input_dim x 1] tensor of inputs to the network.
        batch_u: [fragsize x input_dim x 1] tensor of inputs to the network.
        labels: [fragsize, ] tensor of labels for the inputs.
        param_l: List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u: List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        config: Configuration dictionary.
        label_poison: Boolean flag to indicate if the labels are being poisoned.
    """
    labels = labels.squeeze()
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # get config parameters
    bound_kwargs = config["bound_kwargs"]
    loss = config["loss"]
    forward = config["forward_bound"]
    backward = config["backward_bound"]

    # for a combined forward backward pass, set forward_bound == combined and backward_bound to the function
    if forward == "combined":
        assert label_poison == False, "Combined forward backward pass bounds not supported with label poisoning"
        grad_min, grad_max = agt_definitions.FORWARD_BACKWARD_BOUNDS[backward](
            param_l, param_u, batch_l, batch_u, labels, loss, **bound_kwargs
        )
    else:
        label_epsilon = config["label_epsilon"] if label_poison else 0.0
        k_label_poison = config["label_k_poison"] if label_poison else 0
        poison_target = config["poison_target"] if label_poison else -1
        # forward pass through the network with bounds
        logit_l, logit_u, inter_l, inter_u = agt_definitions.FORWARD_BOUNDS[forward](
            param_l, param_u, batch_l, batch_u, **bound_kwargs
        )
        # calculate the first partial derivative of the loss function
        # (pass logit_u in as a dummy for logit_n and ignore dL_n)
        dL_l, dL_u, _ = agt_definitions.LOSS_BOUNDS[loss](
            logit_l, logit_u, logit_u, labels, k_label_poison=k_label_poison,
            label_epsilon=label_epsilon, poison_target=poison_target
        )
        # compute backwards pass through the network with bounds
        grad_min, grad_max = agt_definitions.BACKWARD_BOUNDS[backward](
            dL_l, dL_u, param_l, param_u, inter_l, inter_u, **bound_kwargs
        )

    return grad_min, grad_max


def break_condition(eval):
    """
    Check whether to terminate the certified training loop based on the bounds on the test metric (MSE or Accuracy).
    eval[0] = worst case eval
    eval[1] = nominal eval
    eval[2] = best case eval
    """
    if eval[0] <= 0.03 and eval[2] >= 0.97:  # worst case accuracy bounds too loose
        logging.warning("Early stopping due to loose bounds")
        return True
    if eval[0] >= 1e2:  # worst case MSE too large
        logging.warning("Early stopping due to loose bounds")
        return True

def validate_config(config):
    """
    Validate the config and set default values.
    """
    config = copy.copy(dict(config))
    # validate and set training parameters
    assert isinstance(config["batchsize"], int) and config["batchsize"] > 0, "batchsize must be a positive integer"
    assert isinstance(config["n_epochs"], int) and config["n_epochs"] > 0, "n_epochs must be a positive integer"
    config.setdefault("interval_matmul", "rump")
    os.environ["INTERVAL_MATMUL"] = config.get("interval_matmul", "rump")
    config.setdefault("bound_kwargs", {})
    config.setdefault("fragsize", config["batchsize"])
    config.setdefault("device", "cpu")
    config.setdefault("optimizer_kwargs", {})
    config.setdefault("fragsize", config["batchsize"])
    assert config.setdefault("optimizer", "sgd") in agt_definitions.OPTIMIZERS.keys(), f"Optimizer must be one of {agt_definitions.OPTIMIZERS.keys()}"
    assert config["loss"] in agt_definitions.LOSS_BOUNDS.keys(), f"Loss function must be one of {agt_definitions.LOSS_BOUNDS.keys()}"
    assert config["backward_bound"] in agt_definitions.BACKWARD_BOUNDS.keys(), f"Backward bound must be one of {agt_definitions.BACKWARD_BOUNDS.keys()}"
    if config["forward_bound"] != "combined":
        assert config["forward_bound"] in agt_definitions.FORWARD_BOUNDS.keys(), f"Forward bound must be one of {agt_definitions.FORWARD_BOUNDS.keys()}"
    assert config.setdefault("learning_rate", 0.01) >= 0, "learning_rate must be non-negative"
    assert config.setdefault("l1_reg", 0.0) >= 0, "l1_reg must be non-negative"
    assert config.setdefault("l2_reg", 0.0) >= 0, "l2_reg must be non-negative"
    # validate and set unlearning parameters
    assert config.setdefault("k_unlearn", 0) >= 0, "k_unlearn must be non-negative"
    # validate and set privacy parameters
    assert config.setdefault("clip_gamma", 1e10) > 0, "clip_gamma must be positive"
    assert config.setdefault("dp_sgd_sigma", 0.0) >= 0, "dp_sgd_sigma must be non-negative"
    assert config.setdefault("k_private", 0) >= 0, "k_private must be non-negative"
    # validate and set poisoning parameters
    assert config.setdefault("k_poison", 0) >= 0, "k_poison must be non-negative"
    assert config.setdefault("epsilon", 0.00) >= 0, "epsilon must be non-negative"
    assert config.setdefault("label_k_poison", 0) >= 0, "label_k_poison must be non-negative"
    assert config.setdefault("label_epsilon", 0.0) >= 0, "label_epsilon must be non-negative"
    assert isinstance(config.setdefault("poison_target", -1), int), "poison_target index must be an integer"
    assert config["k_poison"] + config["label_k_poison"] < config["fragsize"], "k_poison must be <= fragsize"
    # pass warnings for some settings of parameters
    k = config["k_unlearn"] + config["k_private"] + config["k_poison"] + config["label_k_poison"]
    if k == 0:
        logging.warning("k=0 suffers from numerical instability, consider using dtype double or setting k > 0.")
    if config["batchsize"] == 1:
        logging.warning("Training with batchsize=1 not recommended.")
    return config


def get_parameters(model):
    """
    Get the parameters of the dense layers of the pytorch model.
    """
    param_n = [(l.weight, l.bias) for l in model.modules() if isinstance(l, torch.nn.Linear)]  # get linear params
    param_n = [item for sublist in param_n for item in sublist]  # flatten the list
    param_n = [t if len(t.shape) == 2 else t.unsqueeze(-1) for t in param_n]  # reshape bias to [n x 1] instead of [n]
    param_n = [t.detach().clone() for t in param_n]
    param_l = [p.clone() for p in param_n]
    param_u = [p.clone() for p in param_n]
    return param_n, param_l, param_u


def propagate_conv_layers(x, model, epsilon):
    """
    Propagate an input batch through the convolutional layers of a model. Here we assume that the conv layers are all
    at the start of the network with ReLU activations after each one.
    """
    # get the parameters of the conv layers
    conv_layers = [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]
    conv_parameters = [(l.weight.detach(), l.bias.detach(), l.stride, l.padding) for l in conv_layers]
    # propagate the input through the conv layers
    x_l, x_u = x - epsilon, x + epsilon
    for W, b, stride, padding in conv_parameters:
        x_l, x_u = interval_arithmetic.propagate_conv2d(x_l, x_u, W, b, stride, padding)
        x_l, x_u = torch.nn.functional.relu(x_l), torch.nn.functional.relu(x_u)
    x_l = x_l.flatten(start_dim=1)
    x_u = x_u.flatten(start_dim=1)
    return x_l.unsqueeze(-1), x_u.unsqueeze(-1)
