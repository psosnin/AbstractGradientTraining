# %%
import copy
import math

import matplotlib.pyplot as plt

import train_uci
import script_utils

# %%
"""Helper function for plotting the results."""


def plot_training(sweep_results, sweep_label, sweep_vals, ax, nom=False):
    colors = iter(script_utils.palette)

    for val, results in zip(sweep_vals, sweep_results):
        color = next(colors)
        l = rf"${sweep_label}={val}$"
        nominal = [r[1] for r in results[:150]]
        worst = [r[0] if math.isfinite(r[0]) else 10e6 for r in results[:150]]
        best = [r[2] if math.isfinite(r[2]) else 0 for r in results[:150]]
        ax.fill_between(range(len(nominal)), worst, best, color="#ffffff", lw=0)
        ax.fill_between(range(len(nominal)), worst, best, color=color, label=l, lw=0, alpha=0.6)
        if nom:
            ax.plot(nominal, color=color)
    if not nom:
        ax.plot(nominal, color=script_utils.colours["pink"])  # type: ignore
    legend = ax.legend(
        loc="upper right",
        labelspacing=0.15,
        handletextpad=0.5,
        handlelength=0.8,
        borderpad=0.4,
        bbox_to_anchor=[0.96, 1.0],
        fontsize="x-small",
    )
    for t in legend.get_texts():
        t.set_verticalalignment("baseline")


# %%

"""Run sweeps and plot results."""

subplots = (2, 4)
_, _, _, fig_dir = script_utils.make_dirs()

fig, axs = plt.subplots(
    *subplots,
    figsize=script_utils.set_size(1.0, subplots, shrink_height=1.3),
    sharey=True,
    sharex=True,
    layout="constrained",
    dpi=600,
)

fig.supxlabel("Training Iteration", fontsize="small")
fig.supylabel("MSE + Certified Bounds", va="center", fontsize="small")
for ax in axs.flatten():
    ax.set_xlim(-5, 155)
    ax.set_ylim(0, 0.6)

batchsize = 10000
hidden_size = 64
hidden_lay = 1
seed = 15

ks = [10000, 5000, 1000]
config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
config.log_level = "INFO"
config.epsilon = 0.01
results = []
for k in ks:
    config.k_poison = k
    results.append(train_uci.get_training_bounds(config, batchsize, hidden_lay, hidden_size, seed))

print(f"1. Feature poisoning sweep over k for epsilon={config.epsilon}")
plot_training(results, "n", ks, axs[0][0])
axs[0][0].set_title(f"Feature poison $(\\epsilon={config.epsilon})$", fontsize="small", pad=-10)

epsilons = [0.05, 0.02, 0.01]
config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
config.k_poison = 1000
results = []
for epsilon in epsilons:
    config.epsilon = epsilon
    results.append(train_uci.get_training_bounds(config, batchsize, hidden_lay, hidden_size, seed))

print(f"2. Feature poisoning sweep over epsilon for k={config.k_poison}")
plot_training(results, "\\epsilon", epsilons, axs[0][1])
axs[0][1].set_title(f"Feature poison $(n={config.k_poison})$", fontsize="small", pad=-10)

ks = [10000, 5000, 1000]
config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
config.label_epsilon = 0.05
results = []
for label_k_poison in ks:
    config.label_k_poison = label_k_poison
    results.append(train_uci.get_training_bounds(config, batchsize, hidden_lay, hidden_size, seed))

print(f"3. Label poisoning sweep over n for label_epsilon={config.label_epsilon}")
plot_training(results, "n", ks, axs[0][2])
axs[0][2].set_title(rf"Label poison $(\nu={config.label_epsilon})$", fontsize="small", pad=-10)

config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
config.label_k_poison = 1000
epsilons = [0.2, 0.1, 0.05]
results = []
for label_epsilon in epsilons:
    config.label_epsilon = label_epsilon
    results.append(train_uci.get_training_bounds(config, batchsize, hidden_lay, hidden_size, seed))

print(f"4. Label poisoning sweep over label_epsilon for n={config.label_k_poison}")
plot_training(results, r"\nu", epsilons, axs[0][3])
axs[0][3].set_title(f"Label poison $(n={config.label_k_poison})$", fontsize="small", pad=-10)

config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
hls = [3, 2, 1]
config.k_poison = 100
config.epsilon = 0.01
results = []
for hl in hls:
    results.append(train_uci.get_training_bounds(config, batchsize, hl, hidden_size, seed))
print("5. Sweep over hidden layer depth")
plot_training(results, r"d", hls, axs[1][0], True)
axs[1][0].set_title("Depth ($d$)", fontsize="small", pad=-10)

config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
hds = [256, 128, 64]
config.k_poison = 100
config.epsilon = 0.01
results = []
for hd in hds:
    results.append(train_uci.get_training_bounds(config, batchsize, hidden_lay, hd, seed))
print("6. Sweep over hidden layer width")
plot_training(results, r"w", hds, axs[1][1], True)
axs[1][1].set_title("Width ($w$)", fontsize="small", pad=-10)

config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
batchsizes = [100, 1000, 10000]
config.k_poison = 100
config.epsilon = 0.01
results = []
for bs in batchsizes:
    results.append(train_uci.get_training_bounds(config, bs, hidden_lay, hidden_size, seed))
print("7. Sweep over batch size")
plot_training(results, "b", batchsizes, axs[1][2], True)
axs[1][2].set_title("Batch Size $(b)$", fontsize="small", pad=-10)

config = copy.deepcopy(train_uci.NOMINAL_CONFIG)
learning_rates = [0.02, 0.01, 0.005]
config.k_poison = 100
config.epsilon = 0.01
results = []
for lr in learning_rates:
    config.learning_rate = lr
    results.append(train_uci.get_training_bounds(config, batchsize, hidden_lay, hidden_size, seed))
print("8. Sweep over learning rate")
plot_training(results, r"\alpha", learning_rates, axs[1][3], True)
axs[1][3].set_title("Learning Rate $(\\alpha)$", fontsize="small", pad=-10)

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=1.1), dpi=600)
plt.savefig(f"{fig_dir}/uci_training.png")
