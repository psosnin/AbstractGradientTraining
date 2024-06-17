"""
Configuration class for the abstract gradient training module using pydantic dataclasses/
pydantic is a data validation library that uses Python type annotations to validate data.
"""

import logging
from typing import Dict, Callable, Type, Any
import pydantic

from abstract_gradient_training import optimizers
from abstract_gradient_training import bounds
from abstract_gradient_training import loss_gradient_bounds
from abstract_gradient_training import test_metrics

OPTIMIZERS = {
    "sgd": optimizers.SGD,
    "adam": optimizers.ADAM,
}

FORWARD_BOUNDS = {
    "interval": bounds.interval_bound_propagation.bound_forward_pass,
    "crown": bounds.crown.bound_forward_pass,
    "interval+crown": bounds.crown_ibp.bound_forward_pass,
}

BACKWARD_BOUNDS = {
    "interval": bounds.interval_bound_propagation.bound_backward_pass,
    "crown": bounds.crown.bound_backward_pass,
}

LOSS_BOUNDS = {
    "cross_entropy": loss_gradient_bounds.bound_cross_entropy_derivative,
    "binary_cross_entropy": loss_gradient_bounds.bound_bce_derivative,
    "max_margin": loss_gradient_bounds.bound_max_margin_derivative,
    "mse": loss_gradient_bounds.bound_mse_derivative,
    "hinge": loss_gradient_bounds.bound_hinge_derivative,
}


@pydantic.dataclasses.dataclass(config=dict(extra="forbid"))
class AGTConfig:
    """Configuration class for the abstract gradient training module."""

    # optimizer parameters
    n_epochs: int = pydantic.Field(..., gt=0)
    learning_rate: float = pydantic.Field(..., gt=0)
    l1_reg: float = pydantic.Field(0.0, ge=0)
    l2_reg: float = pydantic.Field(0.0, ge=0)
    optimizer: str = pydantic.Field("sgd", json_schema_extra={"in_": OPTIMIZERS.keys()})
    optimizer_kwargs: Dict = pydantic.Field(default_factory=dict)
    loss: str = pydantic.Field(..., json_schema_extra={"in_": LOSS_BOUNDS.keys()})
    device: str = "cpu"
    log_level: str = pydantic.Field(
        "INFO", json_schema_extra={"in_": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}
    )
    # bounding parameters
    forward_bound: str = pydantic.Field("interval", json_schema_extra={"in_": FORWARD_BOUNDS.keys()})
    backward_bound: str = pydantic.Field("interval", json_schema_extra={"in_": BACKWARD_BOUNDS.keys()})
    bound_kwargs: Dict = pydantic.Field(default_factory=dict)
    fragsize: int = pydantic.Field(10000, gt=0)
    # poisoning parameters
    k_poison: int = pydantic.Field(0, ge=0)
    epsilon: float = pydantic.Field(0, ge=0)
    label_k_poison: int = pydantic.Field(0, ge=0)
    label_epsilon: float = pydantic.Field(0, ge=0)
    poison_target: int = pydantic.Field(-1, ge=0)
    # unlearning parameters
    k_unlearn: int = pydantic.Field(0, ge=0)
    # privacy and dp-sgd parameters
    k_private: int = pydantic.Field(0, ge=0)
    clip_gamma: float = pydantic.Field(1e10, gt=0)
    dp_sgd_sigma: float = pydantic.Field(0, ge=0)

    def __post_init__(self):
        k = max(self.k_unlearn, self.k_private, self.k_poison, self.label_k_poison)
        if k == 0:
            logging.warning("k=0 suffers from numerical instability, consider using dtype double or setting k > 0.")
        if self.fragsize <= k:
            raise ValueError(f"fragsize must be greater than k but got fragsize={self.fragsize} and k={k}")

    @pydantic.computed_field
    def optimizer_class(self) -> Type[Any]:
        """Return the optimizer class based on the optimizer name."""
        return OPTIMIZERS[self.optimizer]

    @pydantic.computed_field
    def forward_bound_fn(self) -> Callable:
        """Return the forward bound function based on the forward_bound name."""
        return FORWARD_BOUNDS[self.forward_bound]

    @pydantic.computed_field
    def backward_bound_fn(self) -> Callable:
        """Return the backward bound function based on the backward_bound name."""
        return BACKWARD_BOUNDS[self.backward_bound]

    @pydantic.computed_field
    def loss_bound_fn(self) -> Callable:
        """Return the loss bound function based on the loss name."""
        return LOSS_BOUNDS[self.loss]

    @pydantic.computed_field
    def test_loss_fn(self) -> Callable:
        """Return the test loss function based on the loss name."""
        if self.loss == "mse":
            return test_metrics.test_mse
        else:
            return test_metrics.test_accuracy
