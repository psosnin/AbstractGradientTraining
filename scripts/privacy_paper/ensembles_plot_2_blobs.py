"""
Plots on the blobs dataset for the AGT privacy paper.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import dp_mechanisms_ensembles
import script_utils
import training_utils
import datasets
import models

# %%

config = training_utils.Config(
    learning_rate=2.0,
    n_epochs=1,
    device="cuda:0",
    loss="cross_entropy",
    lr_decay=0.6,
    lr_min=1e-3,
    clip_gamma=0.06,
    k_private=50,
    log_level="WARNING",
    seed=0,
    batchsize=5000,
)

_get_model = lambda seed: models.fully_connected(seed, width=0, depth=0)


ks_privacy = list(range(0, 200, 1))
blobs_pos, blobs_std = 1.25, 0.35
dataset_train_far, dataset_test_far = datasets.get_blobs(blobs_pos, blobs_std, config.batchsize, config.seed)
model = _get_model(config.seed)
model_hash = training_utils.hash_model(model)
cache_name = f"blobs_{model_hash[0:8]}_{config.hash()[0:8]}_{blobs_std}_{blobs_pos}"

config.log_level = "DEBUG"
training_utils.train_with_config_agt(model, config, dataset_train_far, dataset_test=dataset_test_far)
config.log_level = "WARNING"


# %%
"""Sweep over ensemble size, plot result for each mechanism."""

Ts = [5, 25, 50]

ensembles = {}

for T in Ts:
    ensembles[T] = training_utils.train_ensemble_agt(
        T, ks_privacy, _get_model, config, dataset_train_far, cache_name, quiet=False
    )

# %%
"""Get the private accs for each ensemble for the given epsilon."""

epsilons = list(np.logspace(-2, 2, 10))

delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test_far.tensors
test_point, test_label = test_point[:500], test_label[:500]

format_list = lambda x: [f"{e[0]:.2f}" if isinstance(e, tuple) else f"{e:.2f}" for e in x]

global_accs = []
smooth_sens_accs = []
ptr_accs = []

for T in Ts:
    ensemble_agt = ensembles[T]
    ensemble = [a[0] for a in ensemble_agt]
    global_acc = dp_mechanisms_ensembles.predict_global_sens(
        ensemble, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    smooth_sens_acc = dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        ensemble_agt, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    ptr_acc = dp_mechanisms_ensembles.predict_ptr(
        ensemble_agt, test_point, test_label, epsilons, delta=10**-5, n_repeats=n_repeats
    )
    global_accs.append(global_acc)
    smooth_sens_accs.append(smooth_sens_acc)
    ptr_accs.append(ptr_acc)
    print(f"================ {T} ================")
    print(f"epsilons: {format_list(epsilons)}")
    print(f"GS: {format_list(global_acc)}")
    print(f"SS: {format_list(smooth_sens_acc)}")

# %%

subplots = (2, 3)

fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300, sharex=True, sharey=True)

colors = list(script_utils.colours.values())
# colors = list(reversed(script_utils.sequential_colours))[::2][::-1]


[ax.set_xscale("log") for ax in axs.flatten()]

axs[0, 0].set_title("Blobs")
axs[0, 1].set_title("OctMNIST")
axs[0, 2].set_title("IMDB")

for i, T in enumerate(Ts):
    axs[0, 0].plot(epsilons, [g[0] for g in global_accs[i]], label=f"{T=}", color=colors[i])
    axs[1, 0].plot(epsilons, [g[0] for g in smooth_sens_accs[i]], label=f"{T=}", color=colors[i])

axs[1, 0].set_ylabel("Smooth Sens.")
axs[0, 0].set_ylabel("Global Sens.")

fig.supylabel("Accuracy")
fig.supxlabel("Privacy Loss per Query ($\epsilon$)")
axs[0, 0].legend()
axs[1, 0].legend()

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.6), dpi=300)

fig_dir = script_utils.make_dirs()[3]
plt.savefig(f"{fig_dir}/blobs_ensemble_2.pdf", dpi=300)

# %%

subplots = (3, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300, sharex=True, sharey=True)

for ax in axs.flatten():
    ax.set_xscale("log")


for i, T in enumerate(Ts):
    axs[i, 0].set_ylabel("$T=$" + f"{T}", fontsize="small")
    axs[i, 0].plot(
        epsilons,
        [a[0] for a in global_accs[i]],
        label=f"GS",
        color=script_utils.colours["red"],
    )
    axs[i, 0].fill_between(
        epsilons,
        [a[0] - a[1] for a in global_accs[i]],
        [a[0] + a[1] for a in global_accs[i]],
        alpha=0.3,
        color=script_utils.colours["red"],
    )
    axs[i, 0].plot(
        epsilons,
        [a[0] for a in smooth_sens_accs[i]],
        label=f"SS",
        color=script_utils.colours["green"],
    )
    axs[i, 0].fill_between(
        epsilons,
        [a[0] - a[1] for a in smooth_sens_accs[i]],
        [a[0] + a[1] for a in smooth_sens_accs[i]],
        alpha=0.3,
        color=script_utils.colours["green"],
    )

fig.supylabel("Accuracy")
fig.supxlabel("Privacy Loss ($\epsilon$)")
axs[0, 0].legend(loc="lower right")
axs[0, 0].set_title("Blobs")
axs[0, 1].set_title("OctMNIST")
axs[0, 2].set_title("IMDB")

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.8), dpi=300)
