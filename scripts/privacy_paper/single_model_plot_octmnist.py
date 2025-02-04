"""
Plots on the octmnist dataset for the AGT privacy paper.
"""

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import dp_mechanisms
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

dataset_train, dataset_test = datasets.get_octmnist(
    exclude_classes=[0, 1], balanced=True, encode=fixed_conv_layers
)  # a mix of drusen (2) and normal (3)

_, dataset_test_public = datasets.get_octmnist(exclude_classes=[2], encode=fixed_conv_layers)
_, dataset_test_drusen = datasets.get_octmnist(exclude_classes=[0, 1, 3], encode=fixed_conv_layers)
_, dataset_test_all = datasets.get_octmnist(encode=fixed_conv_layers)
ks_privacy = list(range(0, 50, 1)) + list(range(50, 201, 5))
ks_privacy = list(range(0, 10, 1)) + list(range(10, 50, 5)) + list(range(50, 201, 10))
cache_path = f"octmnist_{config.hash()}_{training_utils.hash_model(base_model)}_3"

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

bounded_model_dict = training_utils.sweep_k_values_with_agt(ks_privacy, model, config, dataset_train, cache_path)
regular_model = bounded_model_dict[0]


# %%

noise_free_acc = dp_mechanisms.predict_noise_free(regular_model, *dataset_test_drusen.tensors)
test_point, test_label = dataset_test_drusen.tensors
n_repeats = 10
epsilons = list(np.logspace(-2, 1, 20))

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

axs[0].set_ylabel("Accuracy")
axs[0].set_title("Blobs")
axs[1].set_title("OctMNIST")
axs[2].set_title("IMDB")
for ax in axs:
    ax.set_xlabel(r"Privacy Loss per Query ($\epsilon$)")
    ax.set_xscale("log")

global_accs_means, global_accs_stds = zip(*global_accs)
smooth_accs_means, smooth_accs_stds = zip(*smooth_accs)

axs[1].axhline(noise_free_acc, linestyle="--", color="grey")
axs[1].plot(epsilons, global_accs_means, label="GS", color=script_utils.colours["red"])
axs[1].fill_between(
    epsilons,
    np.array(global_accs_means) - np.array(global_accs_stds),
    np.array(global_accs_means) + np.array(global_accs_stds),
    alpha=0.3,
    color=script_utils.colours["red"],
)
axs[1].plot(epsilons, smooth_accs_means, label="SS", color=script_utils.colours["green"])
axs[1].fill_between(
    epsilons,
    np.array(smooth_accs_means) - np.array(smooth_accs_stds),
    np.array(smooth_accs_means) + np.array(smooth_accs_stds),
    alpha=0.3,
    color=script_utils.colours["green"],
)

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.5), dpi=300)

# %%
results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_single_model.pkl", "wb") as f:
    pickle.dump((epsilons, global_accs, smooth_accs, noise_free_acc), f)
