# %%
import copy

import torch
import torch.utils.data
import sklearn
import sklearn.datasets
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training import AGTConfig

import script_utils

# %%
"""Initialise helper functions. """


def get_dataloaders(train_batchsize, test_batchsize=500, random_state=0, noise=0.1, n_samples=3000, sep=0.2):
    """
    Get dataloaders for the halfmoons dataset using sklearn.datasets, and adding quadratic and cubic features.
    """
    # get the dataset
    x, y = sklearn.datasets.make_moons(
        noise=noise, random_state=random_state, n_samples=train_batchsize + test_batchsize
    )
    # add separation to the moons
    x[y == 0, 1] += sep
    # split the dataset and add polynomial features
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=test_batchsize / (train_batchsize + test_batchsize), random_state=42
    )
    x_train = np.hstack((x_train, x_train**2, x_train**3))
    x_test = np.hstack((x_test, x_test**2, x_test**3))

    print(x_train.shape)
    print(x_test.shape)
    # form dataloaders
    dl_train = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(x_train).double(), torch.from_numpy(y_train)),
        batch_size=train_batchsize,
        shuffle=False,
    )
    dl_test = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(x_test).double(), torch.from_numpy(y_test)),
        batch_size=test_batchsize,
        shuffle=False,
    )
    return dl_train, dl_test


def run_sweep(model, sweep_variable, sweep_values, config, dl_train, dl_test):
    """
    For the config parameter with name sweep_variable, run certified training for sweep_variable set to each value
    in sweep_values.
    """
    config = copy.deepcopy(config)
    results = []
    for v in sweep_values:
        if sweep_variable == "paired_poison":
            config.k_poison = v
            config.label_k_poison = v
        else:
            config.__setattr__(sweep_variable, v)
        bounded_model = IntervalBoundedModel(model)
        agt.poison_certified_training(bounded_model, config, dl_train, dl_test)
        results.append((v, bounded_model))
    return results


# %%
""" Set script parameters."""

SEED = 1234
torch.manual_seed(SEED)
HIDDEN_DIM = 128
BATCHSIZE = 3000
DL_TRAIN, DL_TEST = get_dataloaders(BATCHSIZE)
MODEL = torch.nn.Sequential(
    torch.nn.Linear(6, HIDDEN_DIM),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_DIM, 2),
).double()
NOMINAL_CONFIG = AGTConfig(
    learning_rate=2.0,
    n_epochs=4,
    device="cuda:0",
    loss="cross_entropy",
    lr_decay=0.6,
    lr_min=1e-3,
    log_level="WARNING",
    paired_poison=True,
)
_, _, _, FIG_DIR = script_utils.make_dirs()

CM = mcolors.ListedColormap(script_utils.sequential_colours)
CM_SCATTER = mcolors.ListedColormap(["#F45B69", "#3ABEFF"])
GRIDSIZE = 1000
MARKERSIZE = 6
MARKER_EDGE_WIDTH = 0.3
GRID_PAD = 1.0

conf = copy.deepcopy(NOMINAL_CONFIG)
conf.log_level = "DEBUG"
bounded_model = IntervalBoundedModel(MODEL)
conf.epsilon = 0.01
conf.k_poison = 1
conf.label_k_poison = 1
agt.poison_certified_training(bounded_model, conf, DL_TRAIN, DL_TEST)

# %%
"""Run the sweep for the half-moon dataset."""

# frame 1: feature poisoning sweep over k with fixed epsilon

conf = copy.deepcopy(NOMINAL_CONFIG)
epsilon_1 = 0.01
conf.epsilon = epsilon_1
results_1 = run_sweep(MODEL, "k_poison", [50, 100, 200, 300], conf, DL_TRAIN, DL_TEST)

# frame 2: feature poisoning sweep over epsilon with fixed k
conf = copy.deepcopy(NOMINAL_CONFIG)
k_poison_2 = 10
conf.k_poison = k_poison_2
results_2 = run_sweep(MODEL, "epsilon", [0.05, 0.1, 0.15, 0.2], conf, DL_TRAIN, DL_TEST)

# frame 3: label poisoning sweep over k with fixed epsilon

conf = copy.deepcopy(NOMINAL_CONFIG)
conf.k_poison = 0
conf.epsilon = 0.0
results_3 = run_sweep(MODEL, "label_k_poison", [1, 3, 5, 7], conf, DL_TRAIN, DL_TEST)

# %%

# frame 4: label and feature poisoning sweep over epsilon with fixed k
conf = copy.deepcopy(NOMINAL_CONFIG)
epsilon_4 = 0.1
conf.epsilon = epsilon_4
results_4 = run_sweep(MODEL, "paired_poison", [1, 3, 5, 7], conf, DL_TRAIN, DL_TEST)


# %%

"""Plot the results of the sweeps."""


def plot_sweep(results, sweep_variable_label, ax):
    """
    Given the parameter bounds in the results list returned from run_sweep, compute the worst-case predictions for a
    grid of points covering the half-moon dataset and plot the results as a contour plot.
    """

    # get the bounds of the dataset for plotting
    x_test, y_test = DL_TEST.dataset.tensors  # type: ignore
    x_min, x_max = x_test[:, 0].min() - GRID_PAD, x_test[:, 0].max() + GRID_PAD
    y_min, y_max = x_test[:, 1].min() - GRID_PAD, x_test[:, 1].max() + GRID_PAD

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    # define a grid of points to evaluate the model at
    x = torch.linspace(x_min, x_max, GRIDSIZE)
    y = torch.linspace(y_min, y_max, GRIDSIZE)
    X, Y = torch.meshgrid(x, y)

    # add polynomial features
    in_data = torch.stack(
        (
            X.flatten(),
            Y.flatten(),
            X.flatten() ** 2,
            Y.flatten() ** 2,
            X.flatten() ** 3,
            Y.flatten() ** 3,
        ),
        dim=1,
    )
    in_data = in_data.to(NOMINAL_CONFIG.device).double()

    # compute the nominal model predictions
    bounded_model = results[0][1]
    logits = bounded_model.forward(in_data)
    logits = logits.argmax(dim=1).reshape(GRIDSIZE, GRIDSIZE)
    worst_case_0 = torch.zeros_like(logits)
    worst_case_1 = torch.zeros_like(logits)
    logits = logits.detach().cpu().numpy()
    ax.contour(X, Y, logits, cmap=CM, alpha=1.0, levels=[0.5])

    # for each result in the sweep, compute the worst-case predictions for the entire grid
    for _, bounded_model in results:
        logits_l, logits_u = bounded_model.bound_forward(in_data, in_data)
        worst_case_logits_0 = torch.stack([logits_l[:, 0], logits_u[:, 1]], dim=1)
        worst_case_logits_1 = torch.stack([logits_u[:, 0], logits_l[:, 1]], dim=1)
        worst_case_0 += worst_case_logits_0.argmax(dim=1).reshape(GRIDSIZE, GRIDSIZE)
        worst_case_1 += 1 - worst_case_logits_1.argmax(dim=1).reshape(GRIDSIZE, GRIDSIZE)

    # plot the worst-case predictions
    worst_case = torch.min(worst_case_0, worst_case_1).detach().cpu().numpy()
    boundaries = list(range(-1, len(results) + 1, 1))
    norm = mcolors.Normalize(vmin=worst_case.min(), vmax=worst_case.max())
    ax.contourf(X, Y, worst_case, cmap=CM, norm=norm, levels=boundaries)

    # define custom legend
    custom_lines, custom_labels = [], []
    for i in range(len(results)):
        custom_lines.append(
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                lw=0,
                markerfacecolor=CM(norm(i + 1)),
                color=script_utils.colours["grey"],
                alpha=0.8,
            )
        )
        custom_labels.append(f"${sweep_variable_label}={results[i][0]}$")
    custom_labels = custom_labels[::-1]
    ax.legend(
        custom_lines,
        custom_labels,
        loc="upper right",
        frameon=True,
        labelspacing=0.2,
        handletextpad=0.1,
        borderpad=0.25,
        # fancybox=False,
        # ncols=2,
        # columnspacing=1,
        # edgecolor="black"
        fontsize="x-small",
    )

    # plot the test dataset
    ax.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test,
        edgecolors="k",
        linewidths=MARKER_EDGE_WIDTH,
        s=MARKERSIZE,
        cmap=CM_SCATTER,
    )


subplots = (1, 4)

fig, axs = plt.subplots(*subplots, layout="constrained", sharey=True, dpi=300)

[ax.set_box_aspect(1) for ax in axs]
[ax.set_xticks([]) for ax in axs]
[ax.set_yticks([]) for ax in axs]
axs[0].set_xlabel(f"(a) Features only\n($\epsilon={epsilon_1}, p=\infty, \\nu=0, q=0$)", fontsize="x-small")  # type: ignore
axs[1].set_xlabel(f"(b) Features only\n($n={k_poison_2}, p=\infty, \\nu=0, q=0$)", fontsize="x-small")
axs[2].set_xlabel(f"(c) Labels only\n($n={k_poison_2}, \epsilon=0, p=\infty, \\nu=1, q=0$)", fontsize="x-small")  # type: ignore
axs[3].set_xlabel(f"(d) Labels + features\n ($\epsilon={epsilon_4}, p=\infty, \\nu=1, q=0$)", fontsize="x-small")


plot_sweep(results_1, r"n", axs[0])
plot_sweep(results_2, r"\epsilon", axs[1])
plot_sweep(results_3, r"n", axs[2])
plot_sweep(results_4, r"n", axs[3])

script_utils.apply_figure_size(fig, script_utils.set_size(0.9, subplots, shrink_height=2.2), dpi=300)
plt.savefig(f"{FIG_DIR}/halfmoons.pdf", dpi=300)
