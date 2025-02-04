# %%
"""Run the sweeps for the blobs dataset and plot a heatmap of the smooth sensitivity."""


import torch
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.cm as mcm
import matplotlib.pyplot as plt

from abstract_gradient_training import privacy_utils

import script_utils
import training_utils
import datasets
import models


# %%
"""Set up environment, config, model and datasets"""

config = training_utils.Config(
    learning_rate=2.0,
    n_epochs=4,
    device="cuda:0",
    loss="cross_entropy",
    lr_decay=0.6,
    lr_min=1e-3,
    clip_gamma=0.06,
    k_private=50,
    log_level="WARNING",
    seed=0,
    batchsize=3000,
)

# script parameters
CM_NOMINAL = mcolors.ListedColormap(["#FF0000"])
CM_SCATTER = mcolors.ListedColormap(["#F45B69", "#3ABEFF"])
CM = sns.blend_palette(["#ffffff", "#DDC4DD", "#DCCFEC", "#A997DF", "#4F517D", "#1A3A3A"], n_colors=200, as_cmap=True)

MARKERSIZE = 4
DATASET_TRAIN_CLOSE, DATASET_TEST_CLOSE = datasets.get_blobs(1, 0.75, config.batchsize, config.seed)
DATASET_TRAIN_FAR, DATASET_TEST_FAR = datasets.get_blobs(1.25, 0.35, config.batchsize, config.seed)
GRIDSIZE = 1000
GRIDLIM = 3.8
KS_PRIVACY = list(range(0, 200, 1)) + list(range(200, 1000, 10)) + list(range(1000, 3000, 100))
KS_PRIVACY = list(range(0, 200, 1))

# %%
"""Run all the sweeps."""

model = models.fully_connected(config.seed, width=128, depth=1)
model_hash = training_utils.hash_model(model)

agt_models_close = training_utils.sweep_k_values_with_agt(
    KS_PRIVACY,
    model,
    config,
    DATASET_TRAIN_CLOSE,
    f"blobs_{model_hash[0:8]}_{config.hash()[0:8]}_close",
)
agt_models_far = training_utils.sweep_k_values_with_agt(
    KS_PRIVACY,
    model,
    config,
    DATASET_TRAIN_FAR,
    f"blobs_{model_hash[0:8]}_{config.hash()[0:8]}_far",
)


# %%
"""Compute the smooth sensitivity."""

epsilon = 1.0
delta = 0.0
beta = epsilon / 6

# define a grid of points to evaluate the model at
x = torch.linspace(-GRIDLIM, GRIDLIM, GRIDSIZE)
y = torch.linspace(-GRIDLIM, GRIDLIM, GRIDSIZE)
X, Y = torch.meshgrid(x, y)
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

nominal_model_close = agt_models_close[0]
in_data = in_data.to(nominal_model_close.device).float()

logits = nominal_model_close.forward(in_data)
logits = logits.argmax(dim=1).reshape(GRIDSIZE, GRIDSIZE)
logits_close = logits.detach().cpu().numpy()
smooth_sens_close = privacy_utils.compute_smooth_sensitivity(beta, in_data, agt_models_close)
smooth_sens_close = smooth_sens_close.cpu().numpy().reshape(GRIDSIZE, GRIDSIZE)

nominal_model_far = agt_models_far[0]
logits = nominal_model_far.forward(in_data)
logits = logits.argmax(dim=1).reshape(GRIDSIZE, GRIDSIZE)
logits_far = logits.detach().cpu().numpy()
smooth_sens_far = privacy_utils.compute_smooth_sensitivity(beta, in_data, agt_models_far)
smooth_sens_far = smooth_sens_far.cpu().numpy().reshape(GRIDSIZE, GRIDSIZE)

# %%
"""Plot the smooth sensitivity as a heatmap."""


def plot_sweep(smooth_sens, logits, test_dataset, ax, cm, norm=None):
    # get the whole test dataset
    x_test, y_test = test_dataset.tensors

    # define a grid of points to evaluate the model at
    x = torch.linspace(-GRIDLIM, GRIDLIM, GRIDSIZE)
    y = torch.linspace(-GRIDLIM, GRIDLIM, GRIDSIZE)
    X, Y = torch.meshgrid(x, y)

    ax.contour(X, Y, logits, cmap=CM_NOMINAL, alpha=1.0, levels=[0.5])

    if norm is None:
        norm = mcolors.LogNorm(vmin=smooth_sens.min(), vmax=1.0)

    cont = ax.imshow(np.rot90(smooth_sens), extent=[-GRIDLIM, GRIDLIM, -GRIDLIM, GRIDLIM], norm=norm, cmap=cm)

    ax.scatter(
        x_test[:, 0], x_test[:, 1], s=MARKERSIZE, c=y_test, edgecolors="#c0d1c9", linewidths=0.4, cmap=CM_SCATTER
    )
    ax.scatter(
        x_test[:, 0], x_test[:, 1], s=MARKERSIZE, c=y_test, edgecolors="#c0d1c9", linewidths=0.4, cmap=CM_SCATTER
    )

    return norm, cont


subplots = (1, 2)
fig, axs = plt.subplots(*subplots, dpi=300, sharey=True, layout="constrained")

for ax in axs:
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-GRIDLIM, GRIDLIM)
    ax.set_ylim(-GRIDLIM, GRIDLIM)

norm, cont = plot_sweep(smooth_sens_far, logits_far, DATASET_TEST_FAR, axs[0], CM)
norm, cont = plot_sweep(smooth_sens_close, logits_close, DATASET_TEST_CLOSE, axs[1], CM, norm=norm)
cbar_ax = fig.add_axes((0.25, -0.07, 0.5, 0.03))  # [left, bottom, width, height]

cbar = fig.colorbar(
    mcm.ScalarMappable(cmap=CM, norm=norm),
    cax=cbar_ax,
    orientation="horizontal",
    label="Smooth Sensitivity Bound",
    shrink=1.5,
)
cbar.ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
cbar.ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
cbar.ax.tick_params(length=0)

red_line = mlines.Line2D([], [], color="red", label=r"$f^{\theta}$")
axs[0].legend(
    handles=[red_line],
    loc="lower left",
    handlelength=0.8,
    bbox_to_anchor=(-0.0, -0.0),
    framealpha=1.0,
)

script_utils.apply_figure_size(fig, script_utils.set_size(0.5, subplots, shrink_height=2.1), dpi=300)
plt.savefig(".figures/blobs_privacy_smooth_sens.pdf", dpi=300)

# %%
