"""
Plots on the blobs dataset for the AGT privacy paper.
"""

# %%
import pickle
import numpy as np

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


ks_privacy = (
    list(range(0, 200, 1)) + list(range(200, 500, 20)) + list(range(500, 1001, 50)) + list(range(1000, 2000, 100))
)
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

Ts = list(range(2, 100, 10))
Ts = list(range(1, 10, 1)) + list(range(10, 100, 5))

ensembles = {}

for T in Ts:
    ensembles[T] = training_utils.train_ensemble_agt(
        T, ks_privacy, _get_model, config, dataset_train_far, cache_name, quiet=False
    )

# %%
"""Get the private accs for each ensemble for the given epsilon."""

Ts = [1] + list(range(5, 100, 5))

epsilons = list(np.logspace(-2, 1, 10))
epsilons = [1.0, 0.5, 0.1]

delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test_far.tensors
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
with open(f"{results_dir}/blobs_ensemble.pkl", "wb") as f:
    pickle.dump((Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs), f)

# %%

Ts = [5, 25, 50]

epsilons = list(np.logspace(-2, 1, 10))

delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test_far.tensors
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
with open(f"{results_dir}/blobs_ensemble_2.pkl", "wb") as f:
    pickle.dump((Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs), f)
