"""
Plots on the octmnist dataset for the AGT privacy paper.
"""

# %%
import pickle
import numpy as np

import abstract_gradient_training as agt

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

Ts = [1] + list(range(5, 100, 5))
ensembles = {}

for T in Ts:
    ensembles[T] = training_utils.train_ensemble_agt(
        T, ks_privacy, _get_model, config, dataset_train, cache_path, quiet=False
    )


# %%

epsilons = list(np.logspace(-2, 1, 10))
epsilons = [1.0, 0.5, 0.1]

delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test_drusen.tensors
test_point, test_label = test_point[:500], test_label[:500]

format_list = lambda x: [f"{e[0]:.2f}" if isinstance(e, tuple) else f"{e:.2f}" for e in x]

global_accs = []
smooth_sens_accs = []
noise_free_accs = []

for T in Ts:
    ensemble_agt = ensembles[T]
    ensemble = [a[0] for a in ensemble_agt]
    global_acc = dp_mechanisms_ensembles.predict_global_sens(
        ensemble, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    smooth_sens_acc = dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        ensemble_agt, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    noise_free_acc = dp_mechanisms_ensembles.predict_noise_free(ensemble, test_point, test_label)
    global_accs.append(global_acc)
    smooth_sens_accs.append(smooth_sens_acc)
    noise_free_accs.append(noise_free_acc)
    print(f"================ {T} ================")
    print(f"epsilons: {format_list(epsilons)}")
    print(f"GS: {format_list(global_acc)}")
    print(f"SS: {format_list(smooth_sens_acc)}")

# %%

"""Save the results."""

import pickle

results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_ensemble.pkl", "wb") as f:
    pickle.dump((Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs), f)

# %%

Ts = [5, 25, 50]

epsilons = list(np.logspace(-2, 1, 10))

delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test_drusen.tensors
test_point, test_label = test_point[:500], test_label[:500]

format_list = lambda x: [f"{e[0]:.2f}" if isinstance(e, tuple) else f"{e:.2f}" for e in x]

global_accs = []
smooth_sens_accs = []
noise_free_accs = []

for T in Ts:
    ensemble_agt = ensembles[T]
    ensemble = [a[0] for a in ensemble_agt]
    global_acc = dp_mechanisms_ensembles.predict_global_sens(
        ensemble, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    smooth_sens_acc = dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        ensemble_agt, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    noise_free_acc = dp_mechanisms_ensembles.predict_noise_free(ensemble, test_point, test_label)
    global_accs.append(global_acc)
    smooth_sens_accs.append(smooth_sens_acc)
    noise_free_accs.append(noise_free_acc)
    print(f"================ {T} ================")
    print(f"epsilons: {format_list(epsilons)}")
    print(f"GS: {format_list(global_acc)}")
    print(f"SS: {format_list(smooth_sens_acc)}")

# %%

"""Save the results."""

results_dir = script_utils.make_dirs()[0]
with open(f"{results_dir}/octmnist_ensemble_2.pkl", "wb") as f:
    pickle.dump((Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs), f)
