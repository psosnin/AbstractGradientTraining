"""
Plots on the blobs dataset for the AGT privacy paper.
"""

# %%
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import script_utils

# %%


def plot_results_1(axs, plot_idx, Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs):
    print(epsilons)
    j = 0
    for i, epsilon in enumerate(epsilons):
        if epsilon == 0.5:
            continue

        Ts = [T for T in Ts if T <= 50]
        noise_free_accs = noise_free_accs[: len(Ts)]
        global_accs = global_accs[: len(Ts)]
        smooth_sens_accs = smooth_sens_accs[: len(Ts)]

        axs[j, 0].set_ylabel(r"$\epsilon=$" + f"{epsilon:.2f}", fontsize="small")
        axs[j, plot_idx].plot(
            Ts,
            noise_free_accs,
            color="grey",
            linestyle="--",
            label="Non-private",
        )
        axs[j, plot_idx].plot(
            Ts,
            [a[i][0] for a in global_accs],
            label=f"Subs. & Agg. (GS)",
            color=script_utils.colours["red"],
        )
        axs[j, plot_idx].fill_between(
            Ts,
            [a[i][0] - a[i][1] for a in global_accs],
            [a[i][0] + a[i][1] for a in global_accs],
            alpha=0.3,
            color=script_utils.colours["red"],
        )
        axs[j, plot_idx].plot(
            Ts,
            [a[i][0] for a in smooth_sens_accs],
            label=f"Subs. & Agg. (SS)",
            color=script_utils.colours["green"],
        )
        axs[j, plot_idx].fill_between(
            Ts,
            [a[i][0] - a[i][1] for a in smooth_sens_accs],
            [a[i][0] + a[i][1] for a in smooth_sens_accs],
            alpha=0.3,
            color=script_utils.colours["green"],
        )
        j += 1


# %%
results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_ensemble.pkl", "rb") as f:
    results_octmnist = pickle.load(f)
with open(f"{results_dir}/blobs_ensemble.pkl", "rb") as f:
    results_blobs = pickle.load(f)
with open(f"{results_dir}/imdb_ensemble.pkl", "rb") as f:
    results_imdb = pickle.load(f)

# %%

subplots = (2, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300, sharex=True)

for ax in axs.flatten():
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

plot_results_1(axs, 0, *results_blobs)
plot_results_1(axs, 1, *results_octmnist)
plot_results_1(axs, 2, *results_imdb)

# for ax in axs[:, 0]:
#     ax.set_yticks([1.0, 0.8, 0.6])
#     ax.set_ylim(0.5, 1.05)

# for ax in axs[:, 1]:
#     ax.set_yticks([0.9, 0.7, 0.5])
#     ax.set_ylim(0.5, 0.95)

# for ax in axs[:, 2]:
#     ax.set_yticks([0.8, 0.7, 0.5])
#     ax.set_ylim(0.5, 0.85)

axs[-1, 1].set_xlabel("Ensemble Size $(T)$")

fig.supylabel("Accuracy", fontsize="small")
# fig.supxlabel("Ensemble Size $(T)$", fontsize="medium")
axs[0, 2].legend(loc="lower right", frameon=True, handlelength=1.3, labelspacing=0.3)
axs[0, 0].set_title("Blobs", fontsize="small")
axs[0, 1].set_title("OctMNIST", fontsize="small")
axs[0, 2].set_title("IMDB", fontsize="small")

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.45), dpi=300)

fig_dir = script_utils.make_dirs()[3]
plt.savefig(f"{fig_dir}/ensemble_plot.pdf", dpi=300)


# # %%


# def plot_results_2(axs, plot_idx, Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs):
#     print(epsilons)
#     print(Ts)
#     for i, T in enumerate(Ts):
#         axs[i, plot_idx].set_xscale("log")
#         axs[i, 0].set_ylabel(r"$T=$" + f"{T}", fontsize="small")
#         axs[i, plot_idx].axhline(noise_free_accs[i], color="grey", linestyle="--")
#         axs[i, plot_idx].plot(
#             epsilons,
#             [a[0] for a in global_accs[i]],
#             label=f"GS",
#             color=script_utils.colours["red"],
#         )
#         axs[i, plot_idx].fill_between(
#             epsilons,
#             [a[0] - a[1] for a in global_accs[i]],
#             [a[0] + a[1] for a in global_accs[i]],
#             alpha=0.3,
#             color=script_utils.colours["red"],
#         )
#         axs[i, plot_idx].plot(
#             epsilons,
#             [a[0] for a in smooth_sens_accs[i]],
#             label=f"SS",
#             color=script_utils.colours["green"],
#         )
#         axs[i, plot_idx].fill_between(
#             epsilons,
#             [a[0] - a[1] for a in smooth_sens_accs[i]],
#             [a[0] + a[1] for a in smooth_sens_accs[i]],
#             alpha=0.3,
#             color=script_utils.colours["green"],
#         )


# # %%
# results_dir = script_utils.make_dirs()[0]
# with open(f"{results_dir}/octmnist_ensemble_2.pkl", "rb") as f:
#     results_octmnist = pickle.load(f)
# with open(f"{results_dir}/blobs_ensemble_2.pkl", "rb") as f:
#     results_blobs = pickle.load(f)
# with open(f"{results_dir}/imdb_ensemble_2.pkl", "rb") as f:
#     results_imdb = pickle.load(f)

# # %%

# subplots = (3, 3)
# fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300, sharex=True)

# plot_results_2(axs, 0, *results_blobs)
# plot_results_2(axs, 1, *results_octmnist)
# plot_results_2(axs, 2, *results_imdb)

# fig.supylabel("Accuracy", fontsize="small")
# fig.supxlabel(r"Privacy Loss per Query $(\epsilon)$)", fontsize="small")
# axs[0, 0].legend(loc="lower right")
# axs[0, 0].set_title("Blobs", fontsize="small")
# axs[0, 1].set_title("OctMNIST", fontsize="small")
# axs[0, 2].set_title("IMDB", fontsize="small")

# script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.6), dpi=300)

# fig_dir = script_utils.make_dirs()[3]
# plt.savefig(f"{fig_dir}/ensemble_plot_2.pdf", dpi=300)

# %%
