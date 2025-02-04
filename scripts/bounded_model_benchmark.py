"""Benchmark comparing the tightness and computation time of different bounded model techniques."""

# %%
import time
import logging

import torch
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from torch.utils.data import DataLoader, TensorDataset

import abstract_gradient_training.poisoning
import abstract_gradient_training
from abstract_gradient_training import AGTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel, CROWNBoundedModel, MIPBoundedModel
from abstract_gradient_training.bounded_losses import BoundedCrossEntropyLoss

# %%
# custom logging
LOGGING = False
GUROBI_KWARGS = {}
if LOGGING:
    logger = logging.getLogger("abstract_gradient_training")
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[AGT] [%(levelname)-8s] [%(asctime)s] %(message)s",
    )
    fh = logging.FileHandler("scripts/.logs/agt.log")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    GUROBI_KWARGS = {"LogToConsole": 0, "LogFile": "scripts/.logs/gurobi.log"}

# %%

# Set up model, config and dataset.
SEED = 1234
BATCHSIZE = 2000
torch.manual_seed(SEED)

BOUND_TEST = "forward"  # or "grads"

# Prepare the dataset
X, Y = sklearn.datasets.make_moons(noise=0.1, random_state=0, n_samples=3000)
X[Y == 0, 1] += 0.2
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
X_TRAIN = np.hstack((X_TRAIN, X_TRAIN**2, X_TRAIN**3))
X_TEST = np.hstack((X_TEST, X_TEST**2, X_TEST**3))
DL = DataLoader(
    dataset=TensorDataset(torch.from_numpy(X_TRAIN).float(), torch.from_numpy(Y_TRAIN)),
    batch_size=BATCHSIZE,
    shuffle=False,
)
DL_TEST = DataLoader(
    dataset=TensorDataset(torch.from_numpy(X_TEST).float(), torch.from_numpy(Y_TEST)), batch_size=3, shuffle=False
)


# Initialise the neural network model
class FullyConnected(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay):
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, hidden_dim)]
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


def get_param_bounds(hidden_dim, hidden_lay):
    model = FullyConnected(in_dim=6, out_dim=2, hidden_dim=hidden_dim, hidden_lay=hidden_lay)

    # Train for a few epochs to obtain realistic parameter bounds
    config = AGTConfig(
        learning_rate=2.0,
        l2_reg=0.02,
        n_epochs=2,
        epsilon=0.05,
        k_poison=10,
        device="cuda:0",
        loss="cross_entropy",
        lr_decay=0.6,
        lr_min=1e-3,
        log_level="INFO",
    )
    bounded_model = IntervalBoundedModel(model)
    abstract_gradient_training.poison_certified_training(bounded_model, config, DL, DL_TEST)
    return bounded_model


def gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, starting_bounds, param_interval=True):
    # copy the starting bounds to a new model
    bounded_model = bounded_model_class(starting_bounds.modules)
    bounded_model._param_l = starting_bounds._param_l if param_interval else starting_bounds._param_n
    bounded_model._param_u = starting_bounds._param_u if param_interval else starting_bounds._param_n
    bounded_model._param_n = starting_bounds._param_n

    start = time.time()
    if BOUND_TEST == "forward":
        l, u = bounded_model.bound_forward(batch_l, batch_u)
    elif BOUND_TEST == "grads":
        l, u = bounded_model.bound_backward_combined(
            batch_l, batch_u, labels, BoundedCrossEntropyLoss(reduction="none")
        )
    else:
        raise ValueError(f"Unknown BOUND_TEST: {BOUND_TEST}")
    total_time = time.time() - start
    # calculate norms of the gradient of the first weight matrix W0
    norm = float((u - l).norm())
    return norm, total_time


# Set up bounding method benchmarks
STANDARD_MODEL = get_param_bounds(32, 1)
WIDE_MODEL = get_param_bounds(128, 1)
DEEP_MODEL = get_param_bounds(32, 3)

# %%

# bounding methods to test
BOUNDED_MODELS = {
    "ibp (rump)": IntervalBoundedModel,
    "ibp (exact)": lambda model: IntervalBoundedModel(model, interval_matmul="exact"),
    "ibp (nguyen)": lambda model: IntervalBoundedModel(model, interval_matmul="nguyen"),
    "crown / ibp (rump)": CROWNBoundedModel,
    "crown / ibp (exact)": lambda model: CROWNBoundedModel(model, interval_matmul="exact"),
    "crown (rump)": lambda model: CROWNBoundedModel(model, gradient_bound_mode="linear"),
    "crown (exact)": lambda model: CROWNBoundedModel(model, gradient_bound_mode="linear", interval_matmul="exact"),
    "crown (nguyen)": lambda model: CROWNBoundedModel(model, gradient_bound_mode="linear", interval_matmul="nguyen"),
    "crown (nguyen, zero)": lambda model: CROWNBoundedModel(
        model, gradient_bound_mode="linear", relu_relaxation="zero", interval_matmul="nguyen"
    ),
    "crown (nguyen, one)": lambda model: CROWNBoundedModel(
        model, gradient_bound_mode="linear", relu_relaxation="one", interval_matmul="nguyen"
    ),
    "crown (nguyen, parallel)": lambda model: CROWNBoundedModel(
        model, gradient_bound_mode="linear", relu_relaxation="parallel", interval_matmul="nguyen"
    ),
    "alpha-crown (rump, final)": lambda model: CROWNBoundedModel(
        model, relu_relaxation="optimizable", interval_matmul="rump"
    ),
    "alpha-crown (nguyen, final)": lambda model: CROWNBoundedModel(
        model, relu_relaxation="optimizable", interval_matmul="nguyen"
    ),
    "alpha-crown (exact, final)": lambda model: CROWNBoundedModel(
        model, relu_relaxation="optimizable", interval_matmul="exact"
    ),
    "alpha-crown (exact, full)": lambda model: CROWNBoundedModel(
        model, relu_relaxation="optimizable", interval_matmul="exact", optimize_inter=True
    ),
    "lp (exact, final)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="lp", interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "qcqp (exact, final)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="qcqp", interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "milp (exact, final)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="milp", interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "miqp (exact, final)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="miqp", interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "lp (exact, full)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="lp", optimize_inter=True, interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "qcqp (exact, full)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="qcqp", optimize_inter=True, interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "milp (exact, full)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="milp", optimize_inter=True, interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
    "miqp (exact, full)": lambda model: MIPBoundedModel(
        model, forward_bound_mode="miqp", optimize_inter=True, interval_matmul="exact", gurobi_kwargs=GUROBI_KWARGS
    ),
}


# %%
batch, labels = next(iter(DL_TEST))
batch = batch.to("cuda:0")
labels = labels.to("cuda:0")
epsilon = 0.1
batch_l, batch_u = batch - epsilon, batch + epsilon

# %%

first_col_width = max([len(name) for name in BOUNDED_MODELS]) + 2

print(f"{' Small Model (1x32)':=^{first_col_width + 95}}")
print(f"|{'Bound':^{first_col_width}}|{'Input + Param Interval':^30}|{'Input Interval':^30}|{'Param Interval':^30}|")
print(f"{'':-^125}")

for name, bounded_model_class in BOUNDED_MODELS.items():
    print(f"|{name:^{first_col_width}}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, STANDARD_MODEL, True)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, STANDARD_MODEL, False)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch, batch, labels, STANDARD_MODEL, True)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}|")


# %%

print(f"{' Wide Model (1x128)':=^125}")

print(f"|{'Bound':^{first_col_width}}|{'Input + Param Interval':^30}|{'Input Interval':^30}|{'Param Interval':^30}|")
print(f"{'':-^125}")

for name, bounded_model_class in BOUNDED_MODELS.items():
    print(f"|{name:^{first_col_width}}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, WIDE_MODEL, True)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, WIDE_MODEL, False)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch, batch, labels, WIDE_MODEL, True)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}|")

# %%

print(f"{' Deep Model (3x32)':=^125}")
print(f"|{'Bound':^{first_col_width}}|{'Input + Param Interval':^30}|{'Input Interval':^30}|{'Param Interval':^30}|")
print(f"{'':-^125}")

for name, bounded_model_class in BOUNDED_MODELS.items():
    print(f"|{name:^{first_col_width}}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, DEEP_MODEL, True)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch_l, batch_u, labels, DEEP_MODEL, False)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}", end="")
    n, t = gradient_bound_helper(bounded_model_class, batch, batch, labels, DEEP_MODEL, True)
    print(f"|{f'{n:<6.6g}, {t:>8.2g}s':^30}|")
