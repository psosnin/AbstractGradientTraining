"""
Plots on the blobs dataset for the AGT privacy paper.
"""

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import script_utils

# %%


def plot_results_1(
    axs,
    plot_idx,
    inference_budgets,
    smooth_accs_ensemble,
    smooth_sens_single,
    global_accs_ensemble,
    global_sens_single,
    dp_sgd_acc,
):
    global_accs_ensemble_means, global_accs_ensemble_stds = zip(*global_accs_ensemble)
    smooth_accs_ensemble_means, smooth_accs_ensemble_stds = zip(*smooth_accs_ensemble)
    global_accs_single_means, global_accs_single_stds = zip(*global_sens_single)
    smooth_accs_single_means, smooth_accs_single_stds = zip(*smooth_sens_single)

    axs[plot_idx].axhline(dp_sgd_acc, linestyle="--", color=script_utils.colours["blue"], label="DP-SGD")
    axs[plot_idx].plot(
        inference_budgets, global_accs_ensemble_means, label="Subs. & Agg. (GS)", color=script_utils.colours["red"]
    )
    axs[plot_idx].fill_between(
        inference_budgets,
        np.array(global_accs_ensemble_means) - np.array(global_accs_ensemble_stds),
        np.array(global_accs_ensemble_means) + np.array(global_accs_ensemble_stds),
        alpha=0.3,
        color=script_utils.colours["red"],
    )
    axs[plot_idx].plot(
        inference_budgets,
        global_accs_single_means,
        label="Pred. Sens. (GS)",
        color=script_utils.colours["red"],
        linestyle="--",
    )
    axs[plot_idx].fill_between(
        inference_budgets,
        np.array(global_accs_single_means) - np.array(global_accs_single_stds),
        np.array(global_accs_single_means) + np.array(global_accs_single_stds),
        alpha=0.3,
        color=script_utils.colours["red"],
    )
    axs[plot_idx].plot(
        inference_budgets, smooth_accs_ensemble_means, label="Subs. & Agg. (SS)", color=script_utils.colours["green"]
    )
    axs[plot_idx].fill_between(
        inference_budgets,
        np.array(smooth_accs_ensemble_means) - np.array(smooth_accs_ensemble_stds),
        np.array(smooth_accs_ensemble_means) + np.array(smooth_accs_ensemble_stds),
        alpha=0.3,
        color=script_utils.colours["green"],
    )

    axs[plot_idx].plot(
        inference_budgets,
        smooth_accs_single_means,
        label="Pred. Sens. (SS)",
        color=script_utils.colours["green"],
        linestyle="--",
    )
    axs[plot_idx].fill_between(
        inference_budgets,
        np.array(smooth_accs_single_means) - np.array(smooth_accs_single_stds),
        np.array(smooth_accs_single_means) + np.array(smooth_accs_single_stds),
        alpha=0.3,
        color=script_utils.colours["green"],
    )


# %%
results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_inference_budget.pkl", "rb") as f:
    results_octmnist = pickle.load(f)
with open(f"{results_dir}/blobs_inference_budget.pkl", "rb") as f:
    results_blobs = pickle.load(f)
with open(f"{results_dir}/imdb_inference_budget.pkl", "rb") as f:
    results_imdb = pickle.load(f)

# %%

subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].set_title("Blobs")
axs[1].set_title("OctMNIST")
axs[2].set_title("IMDB")

plot_results_1(axs, 0, *results_blobs)
plot_results_1(axs, 1, *results_octmnist)
plot_results_1(axs, 2, *results_imdb)

for ax in axs:
    ax.set_xscale("log")

axs[-1].legend(handlelength=1.4, loc="center right", bbox_to_anchor=(1.85, 0.5), frameon=True)

axs[0].set_ylabel("Accuracy", fontsize="small")
axs[1].set_xlabel("Inference Budget ($Q$)", fontsize="small")

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.35), dpi=300)

fig_dir = script_utils.make_dirs()[3]
plt.savefig(f"{fig_dir}/inference_budget_plot.pdf", dpi=300)


# %%
