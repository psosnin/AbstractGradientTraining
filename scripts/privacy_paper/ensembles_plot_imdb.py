"""
Plots on the imdb dataset for the AGT privacy paper.
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

ks_privacy = (
    list(range(0, 10, 1))
    + list(range(10, 50, 5))
    + list(range(50, 200, 20))
    + list(range(200, 500, 50))
    + list(range(500, 1001, 100))
)

dataset_train, dataset_test = datasets.get_imdb()

model = _get_model(config.seed)
model_hash = training_utils.hash_model(model)
cache_name = f"blobs_{model_hash[0:8]}_{config.hash()[0:8]}"

config.log_level = "DEBUG"
bounded_model = training_utils.train_with_config_agt(model, config, dataset_train, dataset_test=dataset_test)
config.log_level = "WARNING"

# %%
test_point, test_label = dataset_test.tensors
# idx = np.random.permutation(len(test_point))
# test_point, test_label = test_point[idx[:1000]], test_label[idx[:1000]]
print(test_point.shape)
preds = bounded_model.forward(test_point.to(bounded_model.device)) > 0
print((preds.cpu().squeeze() == test_label.cpu().squeeze()).float().mean().item())

# %%
"""Sweep over ensemble size, plot result for each mechanism."""

Ts = [1] + list(range(5, 100, 5))

ensembles = {}

for T in Ts:
    ensembles[T] = training_utils.train_ensemble_agt(
        T, ks_privacy, _get_model, config, dataset_train, cache_name, quiet=False
    )
    ensemble = [a[0] for a in ensembles[T]]
    print(dp_mechanisms_ensembles.predict_noise_free(ensemble, test_point, test_label))

# %%
"""Get the private accs for each ensemble for the given epsilon."""


epsilons = list(np.logspace(-2, 1, 10))
epsilons = [1.0, 0.5, 0.1]

delta = 10**-5
n_repeats = 10

format_list = lambda x: [f"{e[0]:.2f}" if isinstance(e, tuple) else f"{e:.2f}" for e in x]

global_accs = []
smooth_sens_accs = []
noise_free_accs = []

for T in Ts:
    ensemble_agt = ensembles[T]
    ensemble = [a[0] for a in ensemble_agt]
    noise_free_acc = dp_mechanisms_ensembles.predict_noise_free(ensemble, test_point, test_label)
    global_acc = dp_mechanisms_ensembles.predict_global_sens(
        ensemble, test_point, test_label, epsilons, n_repeats=n_repeats
    )
    smooth_sens_acc = dp_mechanisms_ensembles.predict_smooth_sens_cauchy(
        ensemble_agt, test_point, test_label, epsilons, n_repeats=n_repeats
    )
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
with open(f"{results_dir}/imdb_ensemble.pkl", "wb") as f:
    pickle.dump((Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs), f)


# %%

Ts = [5, 25, 50]

epsilons = list(np.logspace(-2, 1, 10))

delta = 10**-5
n_repeats = 10

test_point, test_label = dataset_test.tensors

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
with open(f"{results_dir}/imdb_ensemble_2.pkl", "wb") as f:
    pickle.dump((Ts, epsilons, global_accs, smooth_sens_accs, noise_free_accs), f)
