"""
Configuration class for the abstract gradient training module using pydantic dataclasses/
pydantic is a data validation library that uses Python type annotations to validate data.
"""

import logging
import hashlib
from typing import Literal
from collections.abc import Callable

import pydantic
import pydantic.json
import torch

from abstract_gradient_training import bounds
from abstract_gradient_training import test_metrics

LOGGER = logging.getLogger(__name__)

FORWARD_BOUNDS = {
    "interval": bounds.interval_bound_propagation.bound_forward_pass,
    "crown": bounds.linear_bound_propagation.bound_forward_pass,
    "interval,crown": bounds.bound_utils.combine_elementwise(
        bounds.interval_bound_propagation.bound_forward_pass, bounds.linear_bound_propagation.bound_forward_pass
    ),
    "miqp": lambda *args: bounds.optimization_bounds.bound_forward_pass(
        *args, relax_binaries=False, relax_bilinear=False
    ),
    "milp": lambda *args: bounds.optimization_bounds.bound_forward_pass(
        *args, relax_binaries=False, relax_bilinear=True
    ),
    "qcqp": lambda *args: bounds.optimization_bounds.bound_forward_pass(
        *args, relax_binaries=True, relax_bilinear=False
    ),
    "lp": lambda *args: bounds.optimization_bounds.bound_forward_pass(*args, relax_binaries=True, relax_bilinear=True),
}

BACKWARD_BOUNDS = {
    "interval": bounds.interval_bound_propagation.bound_backward_pass,
    "crown": bounds.linear_bound_propagation.bound_backward_pass,
    "miqp": lambda *args: bounds.optimization_bounds.bound_backward_pass(
        *args, relax_binaries=False, relax_bilinear=False
    ),
    "qcqp": lambda *args: bounds.optimization_bounds.bound_backward_pass(
        *args, relax_binaries=True, relax_bilinear=False
    ),
}

LOSS_BOUNDS = {
    "cross_entropy": bounds.loss_gradients.bound_cross_entropy_derivative,
    "binary_cross_entropy": bounds.loss_gradients.bound_bce_derivative,
    "max_margin": bounds.loss_gradients.bound_max_margin_derivative,
    "mse": bounds.loss_gradients.bound_mse_derivative,
    "hinge": bounds.loss_gradients.bound_hinge_derivative,
}

# fields that aren't used when serializing and hashing the configuration
EXCLUDE_FIELDS = [
    "fragsize",
    "device",
    "log_level",
    "callback",
    "model_config",
    "forward_bound_fn",
    "backward_bound_fn",
    "loss_bound_fn",
    "test_loss_fn",
    "noise_distribution",
]


class AGTConfig(pydantic.BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Configuration class for the abstract gradient training module."""

    # optimizer parameters
    n_epochs: int = pydantic.Field(..., gt=0, description="Number of epochs to train the model")
    learning_rate: float = pydantic.Field(..., gt=0, description="Initial learning rate")
    l1_reg: float = pydantic.Field(0.0, ge=0, description="L1 regularization factor")
    l2_reg: float = pydantic.Field(0.0, ge=0, description="L2 regularization factor")
    lr_decay: float = pydantic.Field(0.0, ge=0, description="Learning rate decay factor")
    lr_min: float = pydantic.Field(0.0, ge=0, description="Minimum learning rate for learning rate scheduler")
    loss: str = pydantic.Field(..., pattern=f"^({'|'.join(LOSS_BOUNDS.keys())})$", description="Loss function")
    device: str = pydantic.Field("cpu", description="Device to train the model on")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = pydantic.Field(
        "INFO", description="Logging level"
    )
    early_stopping: bool = pydantic.Field(False, description="Whether to use early stopping criterion.")
    # bounding parameters
    forward_bound: str = pydantic.Field(
        "interval", pattern=f"^({'|'.join(FORWARD_BOUNDS.keys())})$", description="Forward pass bounding function"
    )
    backward_bound: str = pydantic.Field(
        "interval", pattern=f"^({'|'.join(BACKWARD_BOUNDS.keys())})$", description="Backward pass bounding function"
    )
    fragsize: int = pydantic.Field(10000, gt=0, description="Size of fragments to split the batch into to avoid OOM")
    # poisoning parameters
    k_poison: int = pydantic.Field(0, ge=0, description="Number of samples whose features can be poisoned")
    epsilon: float = pydantic.Field(0, ge=0, description="Maximum perturbation for feature poisoning")
    label_k_poison: int = pydantic.Field(0, ge=0, description="Number of samples whose labels can be poisoned")
    label_epsilon: float = pydantic.Field(
        0, ge=0, description="Maximum perturbation for label poisoning (in regression settings)"
    )
    poison_target: int = pydantic.Field(
        -1, ge=0, description="If specified, an attacker can only flip labels to this class."
    )
    # unlearning parameters
    k_unlearn: int = pydantic.Field(0, ge=0, description="Number of removals per batch to be certified")
    # privacy and dp-sgd parameters
    k_private: int = pydantic.Field(0, ge=0, description="Number of removals/insertions per batch to be certified")
    clip_gamma: float = pydantic.Field(float("inf"), gt=0, description="Gradient clipping parameter")
    clip_method: Literal["norm", "clamp"] = pydantic.Field("clamp", description="Method for clipping gradients")
    noise_multiplier: float = pydantic.Field(
        0, ge=0, description="Multiplier of the privacy-preserving noise added to gradients."
    )
    noise_type: Literal["gaussian", "laplace"] = pydantic.Field(
        "gaussian", description="Type of privacy-preserving noise to add to gradients"
    )
    metadata: str = pydantic.Field("", description="Additional metadata to store with the configuration")
    callback: Callable = pydantic.Field(lambda *args: None, description="Callback function to call after each epoch")
    bound_kwargs: dict = pydantic.Field(
        default_factory=dict, description="Additional keyword arguments for bounding functions"
    )

    def __post_init__(self):
        k = max(self.k_unlearn, self.k_private, self.k_poison, self.label_k_poison)
        if k == 0:
            LOGGER.warning("k=0 suffers from numerical instability, consider using dtype double or setting k > 0.")

    def hash(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> str:  # pylint: disable=dangerous-default-value
        """Return a hash of the configuration, used for tracking experiments. Should not be used for dynamic storage of
        configurations, as this object is mutable."""
        self_dict = dict(self)
        # drop fields that don't change the results we are trying to store, for example
        for field in drop_fields:
            self_dict.pop(field, None)
        return hashlib.md5(str(self_dict).encode()).hexdigest()

    @pydantic.computed_field
    def forward_bound_fn(self) -> Callable[..., tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Return the forward bound function based on the forward_bound name."""
        return FORWARD_BOUNDS[self.forward_bound]

    @pydantic.computed_field
    def backward_bound_fn(self) -> Callable[..., tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Return the backward bound function based on the backward_bound name."""
        return BACKWARD_BOUNDS[self.backward_bound]

    @pydantic.computed_field
    def loss_bound_fn(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return the loss bound function based on the loss name."""
        return LOSS_BOUNDS[self.loss]

    @pydantic.computed_field
    def test_loss_fn(self) -> Callable[..., tuple[float, float, float]]:
        """Return the test loss function based on the loss name."""
        if self.loss == "mse":
            return test_metrics.test_mse
        return test_metrics.test_accuracy

    @pydantic.computed_field
    def noise_distribution(self) -> Callable[[torch.Size], torch.Tensor]:
        """Return a function to sample the noise distribution based on the noise_type."""
        if self.noise_multiplier == 0:
            return torch.zeros
        if self.clip_gamma == float("inf"):
            raise ValueError(
                f"If clip_gamma is infinite, then noise_multiplier must be 0, but got {self.noise_multiplier}"
            )
        if self.noise_type == "gaussian":
            LOGGER.debug(
                "\tUsing Gaussian privacy-preserving noise (std %.2g)", self.noise_multiplier * self.clip_gamma
            )
            return torch.distributions.Normal(0, self.noise_multiplier * self.clip_gamma).sample
        if self.noise_type == "laplace":
            LOGGER.debug(
                "\tUsing Laplacian privacy-preserving noise (scale %.2g)", self.noise_multiplier * self.clip_gamma
            )
            return torch.distributions.Laplace(0, self.noise_multiplier * self.clip_gamma).sample
        raise ValueError(f"Unknown noise type {self.noise_type}")
