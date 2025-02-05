# Abstract Gradient Training

A package for abstract gradient training of neural networks for certificates of poisoning robustness, machine unlearning and differential privacy.

## Installation

Install the package using pip:

```pip install git+https://github.com/psosnin/AbstractGradientTraining```

## Usage

To train a PyTorch model with abstract gradient training, follow these steps:

1. **Define your model and dataloaders**: Create your PyTorch model and dataloaders as usual.
2. **Wrap the model**: Create a bounded version of your model using the wrappers provided in the `bounded_models` module. This wrapper supports computing bounds over the parameters and gradients of the model.
3. **Configure and train**: Set up the training configuration and initiate training with either privacy, unlearning, or poisoning certification.


```python
import torch
import abstract_gradient_training as agt
# set up the training and validation dataloaders
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=1000)
# set up pytorch model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
# wrap the model
bounded_model = agt.bounded_models.IntervalBoundedModel(model)
# set up configuration object
config = agt.AGTConfig(
    n_epochs=10,
    learning_rate=0.1,
    k_poison=10,
    epsilon=0.01,
    loss="cross_entropy"
)
# train the model using abstract gradient training
agt.poison_certified_training(bounded_model, config, dl_train, dl_val)
# get certified bounds on the logits of the trained model
logits = bounded_model.forward(test_point)
logits_l, logits_u = bounded_model.bound_forward(test_point)
```

Additional tutorials can be found in the `notebooks` directory. Scripts for generating the figures from the papers below can be found in the `scripts` directory.

### Configuration

This package uses a configuration object AGTConfig to pass hyperparameters into the certified training methods. Please
refer to the options provided in `abstract_gradient_training/configuration.py` for a full list of available
hyperparameters.

### GPU memory usage

Abstract gradient training often requires training with large batchsizes. This can lead to out-of-memory errors on GPUs with limited memory. To mitigate this, the package provides a `fragsize` parameter in the configuration object. This parameter controls the size of fragments that each batch is split into before computing bounds over the gradients. This separates physical steps (computing gradients over each batch fragment) and logic steps (applying the certified training update rules) while keeping memory usage low. This logic is handled internally using the `gradient_accumulation` module. Larger values of `fragsize` will reduce the number of fragments per batch and improve performance, but will require more memory. If you encounter out-of-memory errors, try reducing the value of `fragsize`.

### Floating point stability

The implementation of the verification algorithms in this package do not take into account floating point soundness. Under certain conditions, the returned bounds may not be sound due to issues with numerical precision. Any issues will be detected and logged as warnings or errors by the package. If you encounter such a warning, it is recommended to switch to using a double precision data type (e.g. `torch.float64`) for the model parameters and training data. If warnings persist after switching to double precision, this indicates a potential error.

### Batchsize handling

The training batchsize has a significant effect on the tightness of the bounds at each iteration. Therefore, the certified training methods require a fixed batchsize for the entire training process and any incomplete batches are discarded. When using PyTorch dataloaders, this typically results in the last batch per epoch being skipped, which may lead to unexpected behavior.

## Changelog

### 2025-02-05

- Tweaked differential privacy certification.
- Updated scripts for reproducing figures from the papers.
- Simplified handling of physical vs logical batches.
- Fixed GPU memory collection inside the main training loops.

### 2024-12-04

- Support arbitrary shaped PyTorch models and additional modules.
- Change interface to bounding methods via the `bounded_models` module
- Change interface to loss functions and optimizers via the `bounded_losses` and `bounded_optimizers` modules.
- Added optimized linear bound propagation (alpha-CROWN) bounds.
- Added scripts to replicate figures from the papers.

### 2024-10-23

- Added optimization-based bounds using MILP, MIQP, QCQP and LP solvers.
- Added `privacy_utils` module for computing tighter private prediction guarantees.
- Added additional examples for poisoning, unlearning and privacy.
- Changed how fixed convolutional layers are handled by certified training methods.

### Initial release

- Added abstract gradient training methods for poisoning, unlearning and differential privacy.

## References

- [Certified Robustness to Data Poisoning in Gradient-Based Training](https://arxiv.org/pdf/2406.05670v1)
- [Certificates of Differential Privacy and Unlearning for Gradient-Based Training](https://arxiv.org/abs/2406.13433)
