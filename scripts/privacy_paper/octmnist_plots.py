"""
Plots on the octmnist dataset for the AGT privacy paper.
"""

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import dp_mechanisms
import dp_mechanisms_ensembles
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

print(
    "Pretrained acc (Drusen), ",
    agt.test_metrics.test_accuracy(
        agt.bounded_models.IntervalBoundedModel(model), *dataset_test_drusen.tensors, epsilon=0
    )[0],
)
print(
    "Pretrained acc (Public), ",
    agt.test_metrics.test_accuracy(
        agt.bounded_models.IntervalBoundedModel(model), *dataset_test_public.tensors, epsilon=0
    )[0],
)
print(
    "Pretrained acc (All), ",
    agt.test_metrics.test_accuracy(
        agt.bounded_models.IntervalBoundedModel(model), *dataset_test_all.tensors, epsilon=0
    )[0],
)

bounded_model = training_utils.train_with_config_agt(model, config, dataset_train, dataset_test=dataset_test_drusen)

print(
    "Finetuned acc (Drusen), ",
    agt.test_metrics.test_accuracy(bounded_model, *dataset_test_drusen.tensors, epsilon=0)[0],
)
print(
    "Finetuned acc (Public), ",
    agt.test_metrics.test_accuracy(bounded_model, *dataset_test_public.tensors, epsilon=0)[0],
)
print("Finetuned acc (All), ", agt.test_metrics.test_accuracy(bounded_model, *dataset_test_all.tensors, epsilon=0)[0])

# %%
T = 5
bounded_model_dict = training_utils.sweep_k_values_with_agt(ks_privacy, model, config, dataset_train, cache_path)
regular_model = bounded_model_dict[0]
ensemble_agt = training_utils.train_ensemble_agt(T, ks_privacy, _get_model, config, dataset_train, cache_path)

# %%
m = ensemble_agt[0][0]
print(
    "Finetuned acc (Drusen), ",
    agt.test_metrics.test_accuracy(m, *dataset_test_drusen.tensors, epsilon=0)[1],
)
print(
    "Finetuned acc (Public), ",
    agt.test_metrics.test_accuracy(m, *dataset_test_public.tensors, epsilon=0)[1],
)
print("Finetuned acc (All), ", agt.test_metrics.test_accuracy(m, *dataset_test_all.tensors, epsilon=0)[1])
print("Finetuned acc (Drusen + Clean)", agt.test_metrics.test_accuracy(m, *dataset_test.tensors, epsilon=0)[1])

# %%

test_point, test_label = dataset_test_drusen.tensors
n_repeats = 10
epsilon, delta = 1.0, 1e-5
ensemble = [model_dict[0] for model_dict in ensemble_agt]

# noise_free_acc_single_model = dp_mechanisms.predict_noise_free(regular_model, test_point, test_label)
# noise_free_acc_ensemble = dp_mechanisms_ensembles.predict_noise_free(ensemble, test_point, test_label)

mechanisms = {
    "Global Sens, Single Model": lambda epsilon, delta: dp_mechanisms.predict_global_sens(
        regular_model, test_point, test_label, epsilon, n_repeats=n_repeats
    ),
    "Smooth Sens Cauchy, Single Model": lambda epsilon, delta: dp_mechanisms.predict_smooth_sens_cauchy(
        bounded_model_dict, test_point, test_label, epsilon, n_repeats=n_repeats
    ),
    # "PTR+AGT, Single Model": lambda epsilon, delta: dp_mechanisms.predict_ptr(
    #     bounded_model_dict, test_point, test_label, epsilon, delta, n_repeats=n_repeats
    # ),
    "Global Sens, Ensemble": lambda epsilon, delta: dp_mechanisms_ensembles.predict_global_sens(
        ensemble, test_point, test_label, epsilon, n_repeats=n_repeats
    ),
    # "PTR+AGT, Ensemble": lambda epsilon, delta: dp_mechanisms_ensembles.predict_ptr(
    #     ensemble_agt, test_point, test_label, epsilon, delta, n_repeats=n_repeats
    # ),
    "Smooth Sens Cauchy, Ensemble": lambda epsilon, delta: dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        ensemble_agt, test_point, test_label, epsilon, n_repeats=n_repeats
    ),
}

for mechanism_name, mechanism in mechanisms.items():
    print(f"{mechanism_name}: {mechanism(epsilon, delta)}")

# %%

"""Sweep over epsilon, plot result for each mechanism."""

epsilons = list(np.logspace(-2, 2, 20))
results = {mechanism_name: mechanism(epsilons, delta) for mechanism_name, mechanism in mechanisms.items()}

fig, ax = plt.subplots(figsize=(5, 4))


def plot_mean_and_stds(acc_list, label, ax):
    linestyle = "--" if "Ensemble" in label else "-"
    means, stds = zip(*acc_list)
    ax.plot(epsilons, means, label=label, linestyle=linestyle)
    ax.fill_between(epsilons, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)


for mechanism_name, acc_list in results.items():
    plot_mean_and_stds(acc_list, mechanism_name, ax)


ax.legend()
ax.set_xscale("log")
ax.set_xlabel("$\epsilon$ per Query ")
ax.set_ylabel("Accuracy")

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.8), dpi=300)

# %%

"""Sweep over ensemble size, plot result for each mechanism."""

Ts = list(range(3, 70, 10))
Ts = list(range(1, 10))

epsilons = list(np.logspace(-2, 1, 10))
delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test_drusen.tensors
test_point, test_label = test_point[:100], test_label[:100]

format_list = lambda x: [f"{e[0]:.2f}" if isinstance(e, tuple) else f"{e:.2f}" for e in x]

global_accs = []
smooth_sens_accs = []
ptr_accs = []

for T in Ts:
    ensemble_agt = training_utils.train_ensemble_agt(T, ks_privacy, _get_model, config, dataset_train, cache_path)
    ensemble = [a[0] for a in ensemble_agt]
    counts = dp_mechanisms_ensembles.get_ensemble_predictions(ensemble, test_point, test_label)[0]
    preds = counts.argmax(dim=1).to(test_label.device)
    acc = (preds == test_label).float().mean().item()
    stable = dp_mechanisms_ensembles.get_ensemble_stable_distance(ensemble_agt, test_point, counts).mean()
    global_acc = dp_mechanisms_ensembles.predict_global_sens(
        ensemble, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    smooth_sens_acc = dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        ensemble_agt, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    ptr_acc = dp_mechanisms_ensembles.predict_ptr(
        ensemble_agt, test_point, test_label, epsilons, delta, n_repeats=n_repeats
    )
    global_accs.append(global_acc)
    smooth_sens_accs.append(smooth_sens_acc)
    ptr_accs.append(ptr_acc)
    print(f"================ {T} ================")
    print(f"epsilons: {format_list(epsilons)}")
    print(f"Stable distance: {stable:.2f}")
    print(f"Noise-free: {acc:.2f}")
    print(f"GS: {format_list(global_acc)}")
    print(f"SS: {format_list(smooth_sens_acc)}")
    print(f"PTR: {format_list(ptr_acc)}")

# %%

fig, ax = plt.subplots(figsize=(5, 4))


def plot_list(index, Ts, acc_list, label, ax):
    means, stds = zip(*[a[index] for a in acc_list])
    ax.plot(Ts, means, label=label)
    ax.fill_between(Ts, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)


index = 5

plot_list(index, Ts, global_accs, "Global Sensitivity", ax)
# plot_list(index, Ts, smooth_sens_accs, "Smooth Sensitivity", ax)
plot_list(index, Ts, ptr_accs, "PTR", ax)

ax.legend()
ax.set_xlabel("Ensemble Size")
ax.set_ylabel("Private Accuracy")
ax.set_title(f"Epsilon = {epsilons[index]:.2f}")


# %%

results_ss = [
    dp_mechanisms.predict_smooth_sens_cauchy(bounded_model_dict, test_point, test_label, epsilon, n_repeats=n_repeats)
    for epsilon in epsilons
]

results_gs = [
    dp_mechanisms.predict_global_sens(regular_model, test_point, test_label, epsilon, n_repeats=n_repeats)
    for epsilon in epsilons
]
# %%

fig, ax = plt.subplots(figsize=(5, 4))


def plot_list(epsilons, acc_list, label, ax):
    means, stds = zip(*acc_list)
    ax.plot(epsilons, means, label=label)
    ax.fill_between(epsilons, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)


plot_list(epsilons, global_accs[-1], f"GS (T={Ts[-1]})", ax)
plot_list(epsilons, ptr_accs[0], f"PTR (T={Ts[0]})", ax)
plot_list(epsilons, smooth_sens_accs[0], f"SS (T={Ts[0]})", ax)
plot_list(epsilons, results_ss, f"SS (Single Model)", ax)
plot_list(epsilons, results_gs, f"GS (Single Model)", ax)


ax.legend()
ax.set_xlabel("$\epsilon$ per Query ")
ax.set_ylabel("Accuracy")
ax.set_xscale("log")


# %%
