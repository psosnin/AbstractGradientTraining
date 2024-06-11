from abstract_gradient_training.bounds import interval_bound_propagation as ibp
from abstract_gradient_training.bounds import crown
from abstract_gradient_training import loss_gradient_bounds
from abstract_gradient_training import optimizers

"""
Dictionaries of different bounds, losses and optimizers.
"""

FORWARD_BOUNDS = {
    "interval": ibp.bound_forward_pass,
    "crown": crown.bound_forward_pass,
}

LOSS_BOUNDS = {
    "cross_entropy": loss_gradient_bounds.bound_cross_entropy_derivative,
    "binary_cross_entropy": loss_gradient_bounds.bound_bce_derivative,
    "max_margin": loss_gradient_bounds.bound_max_margin_derivative,
    "mse": loss_gradient_bounds.bound_mse_derivative,
    "hinge": loss_gradient_bounds.bound_hinge_derivative,
}

BACKWARD_BOUNDS = {
    "interval": ibp.bound_backward_pass,
    "crown": crown.bound_backward_pass,
}

FORWARD_BACKWARD_BOUNDS = {}

OPTIMIZERS = {
    "sgd": optimizers.SGD,
    "adam": optimizers.ADAM,
    "safe_sgd": optimizers.SafeSGD,
}
