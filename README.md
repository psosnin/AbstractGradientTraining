# Abstract Gradient Training

A package for abstract gradient training of neural networks for certificates of poisoning robustness, machine unlearning and differential privacy.

## Installation

Install the package using pip:

```pip install git+https://github.com/psosnin/AbstractGradientTraining```

## Getting started

To train a `torch.nn.Sequential` model with abstract gradient training, you must set up a dataloader, model and configuration object and then call the corresponding certified training method (poisoning, unlearning or privacy).

```python
import torch
import abstract_gradient_training as agt
# set up dataloaders
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
# set up pytorch model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
# set up configuration object
config = agt.AGTConfig(
    n_epochs=10,
    learning_rate=0.1,
    k_poison=10,
    epsilon=0.01,
    loss="cross_entropy"
)
# run certified training
param_l, param_n, param_u = agt.poison_certified_training(model, config, dl_train, dl_test)
```

Additional usage examples can be found in the `examples` directory.

## Configuration

This package uses a configuration object AGTConfig to pass hyperparameters into the certified training methods. The following table lists the available hyperparameters:

| Parameter         | Type   | Allowed Values                           | Default Value | Description                                                                  |
|-------------------|--------|------------------------------------------|---------------|------------------------------------------------------------------------------|
| `n_epochs`        | int    | > 0                                       | N/A           | Number of epochs for training.                                               |
| `learning_rate`   | float  | > 0                                       | N/A           | Learning rate for the optimizer.                                             |
| `l1_reg`          | float  | >= 0                                      | 0.0           | L1 regularization parameter.                                                 |
| `l2_reg`          | float  | >= 0                                      | 0.0           | L2 regularization parameter.                                                 |
| `lr_decay`          | float  | >= 0                                      | 0.0           | Learning rate decay factor. lr ~ (1 / (1 + decay_rate * batch_no))                                                 |
| `lr_min`          | float  | >= 0                                      | 0.0           | Minimum learning rate for decay scheduler.                                                 |
| `loss`            | str    | "cross_entropy", "binary_cross_entropy", "max_margin", "mse", "hinge" | N/A           | Loss function.                                                               |
| `device`          | str    | Any                                       | "cpu"         | Device for training (e.g., "cpu" or "cuda").                                 |
| `log_level`       | str    | "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" | "INFO"       | Logging level.                                                               |
| `forward_bound`   | str    | "interval", "crown", "interval,crown", "miqp", "milp", "qcqp", "lp"      | "interval"    | Forward bounding method.                                                     |
| `backward_bound`  | str    |"interval", "crown", "miqp", "qcqp" | "interval"    | Backward bounding method.                                                    |
| `bound_kwargs`    | dict   | Any                                       | {}            | Additional keyword arguments for bounding methods. See individual bounding methods for details.                           |
| `fragsize`        | int    | > 0                                       | 10000         | Size of fragments to split each batch into to pass into the bounding methods. Larger is faster but requires more memory.                                          |
| `k_poison`        | int    | >= 0                                      | 0             | **Certified poisoning only** Number of poisoned samples.                                                  |
| `epsilon`         | float  | >= 0                                      | 0.0           | **Certified poisoning only** Epsilon value for poisoning.                                                 |
| `label_k_poison`  | int    | >= 0                                      | 0             | **Certified poisoning only** Number of label-poisoned samples.                                            |
| `label_epsilon`   | float  | >= 0                                      | 0.0           | **Certified poisoning only** Epsilon value for label poisoning.                                           |
| `poison_target`   | int    | >= 0                                      | -1            | **Certified poisoning only** Target index for poisoning.                                                  |
| `k_unlearn`       | int    | >= 0                                      | 0             | **Certified unlearning only** Number of samples to unlearn.                                                |
| `k_private`       | int    | >= 0                                      | 0             | **Certified privacy only** Number of private samples.                                                   |
| `clip_gamma`      | float  | > 0                                       | `inf`         | **Certified privacy and unlearning only** Gradient clipping level.                           |
| `clip_method`      | str  | "norm", "clamp"                            | "clamp"          | **Certified privacy and unlearning only** Type of gradient clipping to use.                           |
| `noise_multiplier`    | float  | >= 0                                      | 0.0           | **Certified privacy and unlearning only** DP-SGD noise multiplier. Added noise has scale `clip_gamma * noise_multiplier`.                               |
| `noise_type`    | str  | "gaussian", "laplace"                                   | "gaussian"           | **Certified privacy and unlearning only** Type of noise to add to gradients.                               |

## Changelog

### 0.1: Initial release

### 0.2 (2024-10-23)

- Added optimization-based bounds using MILP, MIQP, QCQP and LP solvers
- Added `privacy_utils` module for computing tighter private prediction guarantees.
- Added additional examples for poisoning, unlearning and privacy.
- Changed how fixed convolutional layers are handled by certified training methods.

## References

- [Certified Robustness to Data Poisoning in Gradient-Based Training](https://arxiv.org/pdf/2406.05670v1)
- [Certificates of Differential Privacy and Unlearning for Gradient-Based Training](https://arxiv.org/abs/2406.13433)
