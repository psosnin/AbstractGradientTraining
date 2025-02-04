"""Plot accuracy vs inference budget for the octmnist dataset."""

# %%
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt

import dp_mechanisms_ensembles
import dp_mechanisms
import dp_composition
import script_utils
import training_utils
import datasets
import models


# %%

config = training_utils.Config(
    fragsize=1000,
    learning_rate=0.06,
    n_epochs=4,
    device="cuda:1",
    loss="binary_cross_entropy",
    log_level="WARNING",
    optimizer="SGDM",
    optimizer_kwargs={"momentum": 0.95, "nesterov": True},
    lr_decay=0.5,
    clip_gamma=0.9,
    lr_min=0.001,
    batchsize=20000,  # max possible batchsize
    seed=1,
    k_private=10,
)

base_model = models.get_octmnist_pretrained(config.seed, device=config.device)
fixed_conv_layers = base_model[0:5]
model = base_model[5:]
print(model)

dataset_train, dataset_test = datasets.get_octmnist(
    exclude_classes=[0, 1], balanced=True, encode=fixed_conv_layers
)  # a mix of drusen (2) and normal (3)

_, dataset_test_public = datasets.get_octmnist(exclude_classes=[2], encode=fixed_conv_layers)
_, dataset_test_drusen = datasets.get_octmnist(exclude_classes=[0, 1, 3], encode=fixed_conv_layers)
_, dataset_test_all = datasets.get_octmnist(encode=fixed_conv_layers)
ks_privacy = list(range(0, 50, 1)) + list(range(50, 201, 5))
ks_privacy = list(range(0, 10, 1)) + list(range(10, 50, 5)) + list(range(50, 201, 10))
cache_path = f"octmnist_{config.hash()}_{training_utils.hash_model(base_model)}_3"
_get_model = lambda seed: model

# %%

agt_ensemble = training_utils.train_ensemble_agt(25, ks_privacy, _get_model, config, dataset_train, cache_path)
gs_ensemble = [a[0] for a in agt_ensemble]
agt_single_model = training_utils.sweep_k_values_with_agt(ks_privacy, model, config, dataset_train, cache_path)
gs_single_model = agt_single_model[0]

# %%

epsilon_t = 10.0
delta_t = 10**-5
n_repeats = 10
test_point, test_label = dataset_test_drusen.tensors

dp_sgd_config = copy.deepcopy(config)
dp_sgd_config.batchsize = 1000
dp_sgd_model, dp_sgd_epsilon, dp_sgd_delta = training_utils.train_dp_sgd_baseline(
    model, epsilon_t, delta_t, dp_sgd_config, dataset_train, cache_path=cache_path
)
dp_sgd_model = dp_sgd_model.cpu()
print("DP-SGD epsilon:", dp_sgd_epsilon, "DP-SGD delta:", dp_sgd_delta)

preds = (dp_sgd_model(test_point.cpu()) > 0).squeeze()
dp_sgd_acc = (preds == test_label.squeeze()).float().mean().item()
print("DP-SGD accuracy:", dp_sgd_acc)


# %%


def get_accs(inference_budget):
    # first find the privacy budget per query
    epsilon, delta = dp_composition.inverse_composition(
        epsilon_t=epsilon_t, inference_budget=inference_budget, delta_t=delta_t, delta_prime=delta_t
    )
    print(inference_budget, epsilon, delta)

    # point, labels = test_point[:inference_budget], test_label[:inference_budget]
    point, labels = test_point, test_label

    global_accs_ensemble = dp_mechanisms_ensembles.predict_global_sens(
        gs_ensemble, point, labels, epsilon, n_repeats=n_repeats
    )
    smooth_accs_ensemble = dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        agt_ensemble, point, labels, epsilon, n_repeats=n_repeats
    )
    global_sens_single = dp_mechanisms.predict_global_sens(gs_single_model, point, labels, epsilon, n_repeats=n_repeats)
    smooth_sens_single = dp_mechanisms.predict_smooth_sens_cauchy(
        agt_single_model, point, labels, epsilon, n_repeats=n_repeats
    )
    return global_accs_ensemble, smooth_accs_ensemble, global_sens_single, smooth_sens_single


inference_budgets = list(int(a) for a in np.logspace(0, 3, 10))
inference_budgets.pop(0)

global_accs_ensemble = []
smooth_accs_ensemble = []
global_sens_single = []
smooth_sens_single = []

for inference_budget in inference_budgets:
    a, b, c, d = get_accs(inference_budget)
    global_accs_ensemble.append(a)
    smooth_accs_ensemble.append(b)
    global_sens_single.append(c)
    smooth_sens_single.append(d)

# %%

subplots = (1, 3)
fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300, sharey=True)

axs[0].set_title("Blobs")
axs[1].set_title("OctMNIST")
axs[2].set_title("IMDB")

plot_idx = 1

for ax in axs:
    ax.set_xscale("log")

fig.supylabel("Accuracy")
fig.supxlabel("Inference Budget ($Q$)")

global_accs_ensemble_means, global_accs_ensemble_stds = zip(*global_accs_ensemble)
smooth_accs_ensemble_means, smooth_accs_ensemble_stds = zip(*smooth_accs_ensemble)
global_accs_single_means, global_accs_single_stds = zip(*global_sens_single)
smooth_accs_single_means, smooth_accs_single_stds = zip(*smooth_sens_single)

axs[plot_idx].axhline(dp_sgd_acc, linestyle="--", color="grey", label="DP-SGD")
axs[plot_idx].plot(
    inference_budgets, global_accs_ensemble_means, label="GS ($T=25$)", color=script_utils.colours["red"]
)
axs[plot_idx].fill_between(
    inference_budgets,
    np.array(global_accs_ensemble_means) - np.array(global_accs_ensemble_stds),
    np.array(global_accs_ensemble_means) + np.array(global_accs_ensemble_stds),
    alpha=0.3,
    color=script_utils.colours["red"],
)
axs[plot_idx].plot(
    inference_budgets, smooth_accs_ensemble_means, label="SS ($T=25$)", color=script_utils.colours["green"]
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
    global_accs_single_means,
    label="GS ($T=1$)",
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
    inference_budgets,
    smooth_accs_single_means,
    label="SS ($T=1$)",
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

axs[plot_idx].legend()
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.5), dpi=300)

# %%

"""Save the results."""

results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_inference_budget.pkl", "wb") as f:
    pickle.dump(
        (
            inference_budgets,
            smooth_accs_ensemble,
            smooth_sens_single,
            global_accs_ensemble,
            global_sens_single,
            dp_sgd_acc,
        ),
        f,
    )
