"""Helper functions for unit tests."""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import uci_datasets  # python -m pip install git+https://github.com/treforevans/uci_datasets.git

from abstract_gradient_training.bounds import loss_gradients
from abstract_gradient_training import configuration


SHAPES_CLASSIFICATION = [
    (3, 6, 5, 3),  # two hidden layers
    (3, 5, 3),  # one hidden layer
]
SHAPES_BINARY_CLASSIFICAITON = [
    (3, 5, 3, 1),  # two hidden layers
    (3, 5, 1),  # one hidden layer
]
SHAPES_REGRESSION = [
    (3, 5, 3, 1),  # two hidden layers
    (3, 5, 1),  # one hidden layer
]
SHAPES_ALL = SHAPES_CLASSIFICATION + SHAPES_BINARY_CLASSIFICAITON + SHAPES_REGRESSION
EPSILONS = [1.0, 0.1, 0.01]
BATCHSIZES = [100]
N_SEEDS = 1
TOLERANCE = 1e-8
FORWARD_BOUNDS = configuration.FORWARD_BOUNDS.values()
BACKWARD_BOUNDS = configuration.BACKWARD_BOUNDS.values()
LOSS_FNS_CLASSIFICATION = [
    (F.cross_entropy, loss_gradients.bound_cross_entropy_derivative),
    (F.multi_margin_loss, loss_gradients.bound_max_margin_derivative),
]
LOSS_FNS_BINARY_CLASSIFICATION = [
    (F.hinge_embedding_loss, loss_gradients.bound_hinge_derivative),
    (F.binary_cross_entropy_with_logits, loss_gradients.bound_bce_derivative),
]
LOSS_FNS_REGRESSION = [(F.mse_loss, loss_gradients.bound_mse_derivative)]
NOMINAL_CONFIG = configuration.AGTConfig(
    n_epochs=2,
    learning_rate=0.1,
    loss="mse",
    device="cuda",
    log_level="DEBUG",
    early_stopping=True,
    clip_gamma=10,
)
K_POISON = [0, 1]
K_UNLEARN = [0, 1]
K_PRIVATE = [0, 1]


def validate_sound(lower: list[torch.Tensor] | torch.Tensor, upper: list[torch.Tensor] | torch.Tensor):
    """
    Validate for the two lists of tensors that the lower bound is less than or equal to the upper bound.

    Args:
        lower (list[torch.Tensor] | torch.Tensor): tensor or list of tensors representing the lower bound
        upper (list[torch.Tensor] | torch.Tensor): tensor or list of tensors representing the upper bound
    """
    lower = [lower] if not isinstance(lower, list) else lower
    upper = [upper] if not isinstance(upper, list) else upper
    assert len(lower) == len(upper)
    for l, u in zip(lower, upper):
        assert l.shape == u.shape, f"{l.shape}, {u.shape}"
        assert l.dtype == u.dtype, f"{l.dtype}, {u.dtype}"
        assert torch.all(l <= u + TOLERANCE), f"{torch.max(l - u)}"


def validate_equal(lower: list[torch.Tensor] | torch.Tensor, upper: list[torch.Tensor] | torch.Tensor):
    """
    Validate for the two lists of tensors are equal.

    Args:
        lower (list[torch.Tensor] | torch.Tensor): tensor or list of tensors representing the lower bound
        upper (list[torch.Tensor] | torch.Tensor): tensor or list of tensors representing the upper bound
    """
    lower = [lower] if not isinstance(lower, list) else lower
    upper = [upper] if not isinstance(upper, list) else upper
    assert len(lower) == len(upper)
    for l, u in zip(lower, upper):
        assert l.shape == u.shape, f"{l.shape}, {u.shape}"
        assert l.dtype == u.dtype, f"{l.dtype}, {u.dtype}"
        assert torch.allclose(l, u), f"{torch.max(torch.abs(l - u))}"


def generate_network(shape: list[int], seed: int):
    """
    Generate the parameters of a neural network with the given shape plus bounds.

    Args:
        shape (list[int]): list of integers representing the shape of the network
        seed (int): seed for random number generation
    """
    # set the seed and initialise containers
    torch.random.manual_seed(seed)
    param_n = []
    param_l = []
    param_u = []

    # generate the parameters
    for i in range(len(shape) - 1):
        W = torch.randn(shape[i + 1], shape[i], requires_grad=True).double()
        b = torch.randn(shape[i + 1], 1, requires_grad=True).double()
        param_n.append(W)
        param_n.append(b)
        param_l.append(W - torch.rand(W.shape))
        param_l.append(b - torch.rand(b.shape))
        param_u.append(W + torch.rand(W.shape))
        param_u.append(b + torch.rand(b.shape))

    return param_l, param_n, param_u


class FullyConnected(torch.nn.Sequential):
    """Example fully connected neural network for tests."""

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay):
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(hidden_lay):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


def get_dataloaders(train_batchsize, test_batchsize=500):
    """
    Get dataloaders for the uci datasets.
    """
    # Get the dataset
    data = uci_datasets.Dataset("forest")
    x_train, y_train, x_test, y_test = data.get_split(split=0)

    # Convert to float32 and add an extra dimension
    x_train, y_train = x_train.astype("float32")[:, :, None], y_train.astype("float32")[:, :, None]
    x_test, y_test = x_test.astype("float32")[:, :, None], y_test.astype("float32")[:, :, None]

    # Normalise x_train, x_test, y_train, y_test
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

    # Convert to tensors and set datatypes
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Form datasets and dataloaders
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=train_batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batchsize, shuffle=True)

    return train_loader, test_loader
