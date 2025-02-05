"""
Configuration class for the abstract gradient training module using pydantic dataclasses/
pydantic is a data validation library that uses Python type annotations to validate data.
The configuration class also provides some QOL logging functions for logging AGT runs.
"""

import logging
import hashlib
from typing import Literal
from collections.abc import Callable

import pydantic
import pydantic.json
import torch

from abstract_gradient_training import bounded_losses
from abstract_gradient_training import bounded_models
from abstract_gradient_training import bounded_optimizers

LOGGER = logging.getLogger(__name__)

LOSS_FNS = ["mse", "binary_cross_entropy", "cross_entropy"]

# fields that aren't used when serializing and hashing the configuration
EXCLUDE_FIELDS = [
    "fragsize",
    "device",
    "log_level",
    "early_stopping_callback",
    "on_iter_end_callback",
    "on_iter_start_callback",
    "model_config",
    "bounded_loss_fn",
    "val_loss",
    "bounded_optimizer_fn",
]


class AGTConfig(pydantic.BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Configuration class for the abstract gradient training module."""

    # optimizer parameters
    n_epochs: int = pydantic.Field(..., gt=0, description="Number of epochs to train the model")
    optimizer: str = pydantic.Field("SGD", description="Optimizer to use for training")
    optimizer_kwargs: dict = pydantic.Field({}, description="Additional keyword arguments for the optimizer")
    learning_rate: float = pydantic.Field(..., gt=0, description="Initial learning rate")
    l1_reg: float = pydantic.Field(0.0, ge=0, description="L1 regularization factor")
    l2_reg: float = pydantic.Field(0.0, ge=0, description="L2 regularization factor")
    lr_decay: float = pydantic.Field(0.0, ge=0, description="Learning rate decay factor")
    lr_min: float = pydantic.Field(0.0, ge=0, description="Minimum learning rate for learning rate scheduler")
    loss: str = pydantic.Field(..., pattern=f"^({'|'.join(LOSS_FNS)})$", description="Loss function")
    device: str = pydantic.Field("cpu", description="Device to train the model on")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = pydantic.Field(
        "INFO", description="Logging level"
    )
    fragsize: int = pydantic.Field(10000, gt=0, description="Size of fragments to split the batch into to avoid OOM")
    # poisoning parameters
    k_poison: int = pydantic.Field(0, ge=0, description="Number of samples whose features can be poisoned")
    epsilon: float = pydantic.Field(0, ge=0, description="Maximum perturbation for feature poisoning")
    label_k_poison: int = pydantic.Field(0, ge=0, description="Number of samples whose labels can be poisoned")
    label_epsilon: float = pydantic.Field(
        0, ge=0, description="Maximum perturbation for label poisoning (in regression settings)"
    )
    poison_target_idx: int = pydantic.Field(
        -1, ge=0, description="If specified, an attacker can only flip labels to this class."
    )
    paired_poison: bool = pydantic.Field(
        False, description="Whether the label and feature poisoning are paired (i.e. the same samples are poisoned)."
    )
    # unlearning parameters
    k_unlearn: int = pydantic.Field(0, ge=0, description="Number of removals per batch to be certified")
    # privacy and dp-sgd parameters
    k_private: int = pydantic.Field(0, ge=0, description="Number of removals/insertions per batch to be certified")
    clip_gamma: float = pydantic.Field(float("inf"), gt=0, description="Gradient clipping parameter")
    clip_method: Literal["norm", "clamp"] = pydantic.Field("clamp", description="Method for clipping gradients")
    metadata: str = pydantic.Field("", description="Additional metadata to store with the configuration")

    # callback functions
    early_stopping_callback: Callable = pydantic.Field(
        lambda *args: False,
        description="Callback function that takes a BoundedModel and returns True if training should terminate.",
    )
    on_iter_end_callback: Callable = pydantic.Field(
        lambda *args: None,
        description="Callback function that takes a BoundedModel and is called at the end of each iteration.",
    )
    on_iter_start_callback: Callable = pydantic.Field(
        lambda *args: None,
        description="Callback function that takes a BoundedModel and is called at the start of each iteration.",
    )

    @property
    def val_loss(self) -> str:
        """
        Return the name of the validation loss function, which is mse for regression and accuracy for classification.
        """
        if self.loss == "mse":
            return "mse"
        return "accuracy"

    def get_bounded_loss_fn(self) -> bounded_losses.BoundedLoss:
        """Return the bounded loss function."""
        if self.loss == "mse":
            return bounded_losses.BoundedMSELoss(reduction="none")
        elif self.loss == "binary_cross_entropy":
            return bounded_losses.BoundedBCEWithLogitsLoss(reduction="none")
        elif self.loss == "cross_entropy":
            return bounded_losses.BoundedCrossEntropyLoss(reduction="none")
        raise ValueError(f"Unknown loss function {self.loss}")

    def get_val_loss_fn(self) -> bounded_losses.BoundedLoss:
        """Return the bounded validation loss function."""
        if self.loss == "mse":
            return bounded_losses.BoundedMSELoss(reduction="mean")
        return bounded_losses.BoundedAccuracy(reduction="mean")

    def get_bounded_optimizer(self, model: bounded_models.BoundedModel) -> bounded_optimizers.BoundedOptimizer:
        """Return the bounded optimizer function."""
        optimizer_kwargs = dict(self.optimizer_kwargs)
        # add the standard optimizer kwargs to the optimizer_kwargs dict
        keys = "learning_rate", "l1_reg", "l2_reg", "lr_decay", "lr_min"
        for key in keys:
            if key in optimizer_kwargs:
                raise ValueError(f"{key} in optimizer_kwargs when it is a default config value.")
            optimizer_kwargs[key] = getattr(self, key)
        if self.optimizer == "SGD":
            return bounded_optimizers.BoundedSGD(model, **optimizer_kwargs)
        elif self.optimizer == "SGDM":
            return bounded_optimizers.BoundedSGDM(model, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

    def hash(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> str:
        """Return a hash of the configuration, used for tracking experiments. Should not be used for dynamic storage of
        configurations, as this object is mutable."""
        self_dict = self.asdict(drop_fields)
        return hashlib.md5(str(self_dict).encode()).hexdigest()

    def asdict(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> dict:
        """Return the configuration as a dictionary, excluding the fields in drop_fields."""
        self_dict = dict(self)
        # drop fields that don't change the results we are trying to store, for example
        for field in drop_fields:
            self_dict.pop(field, None)
        return self_dict
