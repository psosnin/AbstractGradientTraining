"""
Plots on the blobs dataset for the AGT privacy paper.
"""

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

import dp_mechanisms
import script_utils
import training_utils
import datasets
import models

# %%

config = training_utils.Config(
    fragsize=5000,
    learning_rate=0.2,
    n_epochs=3,
    device="cuda:0",
    clip_gamma=0.04,
    loss="binary_cross_entropy",
    log_level="WARNING",
    lr_decay=0.5,
    seed=123,
    batchsize=100000,
    k_private=50,
    optimizer="SGDM",
    optimizer_kwargs={"momentum": 0.99, "dampening": 0.3},
)

_get_model = lambda seed: models.fully_connected(seed, width=100, depth=1, in_dim=768, out_dim=1)

ks_privacy = list(range(0, 50, 1)) + list(range(50, 201, 10))
dataset_train, dataset_test = datasets.get_imdb()

model = _get_model(config.seed)
model_hash = training_utils.hash_model(model)
cache_name = f"blobs_{model_hash[0:8]}_{config.hash()[0:8]}"

bounded_model = training_utils.train_with_config_agt(model, config, dataset_train, dataset_test=dataset_test)

# %%

bounded_model_dict = training_utils.sweep_k_values_with_agt(ks_privacy, model, config, dataset_train, cache_name)
regular_model = bounded_model_dict[0]

# %%

test_point, test_label = dataset_test.tensors
n_repeats = 10
epsilons = list(np.logspace(-2, 1, 20))

noise_free_acc = dp_mechanisms.predict_noise_free(regular_model, test_point, test_label)
global_sens_fn = lambda epsilon: dp_mechanisms.predict_global_sens(
    regular_model, test_point, test_label, epsilon, n_repeats=n_repeats
)
smooth_sens_cauchy_fn = lambda epsilon: dp_mechanisms.predict_smooth_sens_cauchy(
    bounded_model_dict, test_point, test_label, epsilon, n_repeats=n_repeats
)

global_accs = global_sens_fn(epsilons)
smooth_accs = smooth_sens_cauchy_fn(epsilons)


# %%
"""Plot the results."""

subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300, sharey=True)

plot_idx = 2

axs[0].set_ylabel("Accuracy")
axs[0].set_title("Blobs")
axs[1].set_title("OctMNIST")
axs[2].set_title("IMDB")
for ax in axs:
    ax.set_xlabel(r"Privacy Loss per Query ($\epsilon$)")
    ax.set_xscale("log")

global_accs_means, global_accs_stds = zip(*global_accs)
smooth_accs_means, smooth_accs_stds = zip(*smooth_accs)

axs[plot_idx].axhline(noise_free_acc, linestyle="--", color="grey")
axs[plot_idx].plot(epsilons, global_accs_means, label="GS", color=script_utils.colours["red"])
axs[plot_idx].fill_between(
    epsilons,
    np.array(global_accs_means) - np.array(global_accs_stds),
    np.array(global_accs_means) + np.array(global_accs_stds),
    alpha=0.3,
    color=script_utils.colours["red"],
)
axs[plot_idx].plot(epsilons, smooth_accs_means, label="SS", color=script_utils.colours["green"])
axs[plot_idx].fill_between(
    epsilons,
    np.array(smooth_accs_means) - np.array(smooth_accs_stds),
    np.array(smooth_accs_means) + np.array(smooth_accs_stds),
    alpha=0.3,
    color=script_utils.colours["green"],
)
axs[plot_idx].legend()
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.5), dpi=300)


# %%

results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/imdb_single_model.pkl", "wb") as f:
    pickle.dump((epsilons, global_accs, smooth_accs, noise_free_acc), f)
