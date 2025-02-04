"""
Plots on the blobs dataset for the AGT privacy paper.
"""

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import script_utils

# %%


def plot_results(axs, plot_idx, epsilons, global_accs, smooth_accs, noise_free_acc):
    global_accs_means, global_accs_stds = zip(*global_accs)
    smooth_accs_means, smooth_accs_stds = zip(*smooth_accs)

    axs[plot_idx].axhline(noise_free_acc, linestyle="--", color="grey", label="Non-private")
    axs[plot_idx].plot(epsilons, global_accs_means, label="Pred. Sens. (GS)", color=script_utils.colours["red"])
    axs[plot_idx].fill_between(
        epsilons,
        np.array(global_accs_means) - np.array(global_accs_stds),
        np.array(global_accs_means) + np.array(global_accs_stds),
        alpha=0.3,
        color=script_utils.colours["red"],
    )
    axs[plot_idx].plot(epsilons, smooth_accs_means, label="Pred. Sens. (SS)", color=script_utils.colours["green"])
    axs[plot_idx].fill_between(
        epsilons,
        np.array(smooth_accs_means) - np.array(smooth_accs_stds),
        np.array(smooth_accs_means) + np.array(smooth_accs_stds),
        alpha=0.3,
        color=script_utils.colours["green"],
    )


# %%
results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_single_model.pkl", "rb") as f:
    results_octmnist = pickle.load(f)
with open(f"{results_dir}/blobs_single_model.pkl", "rb") as f:
    results_blobs = pickle.load(f)
with open(f"{results_dir}/imdb_single_model.pkl", "rb") as f:
    results_imdb = pickle.load(f)

# %%

subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].set_title("Blobs")
axs[1].set_title("OctMNIST")
axs[2].set_title("IMDB")
for ax in axs:
    ax.set_xscale("log")

plot_results(axs, 0, *results_blobs)
plot_results(axs, 1, *results_octmnist)
plot_results(axs, 2, *results_imdb)
axs[2].legend(handlelength=1.2, loc="lower right", frameon=True, labelspacing=0.3)

axs[0].set_ylabel("Accuracy")
axs[1].set_xlabel(r"Privacy Loss per Query ($\epsilon$)")
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.35), dpi=300)

fig_dir = script_utils.make_dirs()[3]
plt.savefig(f"{fig_dir}/single_model_plot.pdf", dpi=300)
