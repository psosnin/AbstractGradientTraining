"""
Differentially private mechanisms for ensembles, for the privacy paper.

Since we are using these only for generating plots, we allow each prediction mechanism to
take in a list of epsilons and also perform multiple runs so that we can average the noisy accuracies.
Performing predictions like this is not safe for real-world privacy guarantees, its just faster to re-use intermediate
results for generating plots.
"""

# %%

from collections.abc import Callable, Sequence
import numpy as np
import torch
from abstract_gradient_training import privacy_utils
from abstract_gradient_training.bounded_models import BoundedModel


def get_ensemble_predictions(
    ensemble: Sequence[torch.nn.Sequential | BoundedModel],
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the predictions of the ensemble on a batch of data. Returns a (N, 2) tensor of counts for each class.
    """
    model = ensemble[0]

    if isinstance(model, BoundedModel):
        device = torch.device(model.device) if model.device != -1 else torch.device("cpu")
    else:
        device = torch.device(model.parameters().__next__().device)

    batch, labels = batch.to(device), labels.to(device)

    predictions = []

    for model in ensemble:
        logits = model.forward(batch)
        if logits.size(1) == 1:  # binary classification with 1 output logit
            pred = torch.round(torch.sigmoid(logits)).squeeze()
        elif logits.size(1) == 2:  # binary classification with 2 output logits
            pred = torch.argmax(logits, dim=1).squeeze()
        else:
            raise ValueError("Only binary classification is supported.")
        predictions.append(pred)

    predictions = torch.stack(predictions, dim=1)

    n_classes = 2
    counts = torch.zeros(predictions.size(0), n_classes, device=device)
    for i in range(n_classes):
        counts[:, i] = (predictions == i).sum(dim=1)

    return counts, labels


def handle_repeats(
    f: Callable, epsilon: float | list[float], n_repeats: int
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Call f on each epsilon in the list, repeat n_repeats times and
    return the (mean, std) or list of (means, stds).
    """
    if isinstance(epsilon, list):
        accuracies = []
        for eps in epsilon:
            accs = [f(eps) for _ in range(n_repeats)]
            accuracies.append((np.mean(accs), np.std(accs)))
        return accuracies

    accs = [f(epsilon) for _ in range(n_repeats)]
    return float(np.mean(accs)), float(np.std(accs))


def predict_noise_free(
    ensemble: Sequence[torch.nn.Sequential | BoundedModel], batch: torch.Tensor, labels: torch.Tensor
) -> float:
    """
    Compute the accuracy of the ensemble on a batch of data.
    """
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)
    return (counts.argmax(dim=1) == labels).float().mean().item()


def predict_global_sens(
    ensemble: Sequence[torch.nn.Sequential | BoundedModel],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the model on a batch of data with the Laplace mechanism and the global
    sensitivity of the prediction.

    The mechanism is (epsilon, 0)-differentially private.

    """
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)

    def private_prediction(eps: float) -> float:
        noise_distribution = torch.distributions.laplace.Laplace(0, 2.0 / eps + 1e-8)
        noise = noise_distribution.sample(counts.shape).to(counts.device)
        counts_private = counts + noise
        private_predictions = counts_private.argmax(dim=1)
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


def predict_ptr_without_agt(
    ensemble: Sequence[torch.nn.Sequential | BoundedModel],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    delta: float,
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the model on a batch of data with the PTR mechanism.
    """
    # epsilon is halved for each step

    counts, labels = get_ensemble_predictions(ensemble, batch, labels)
    predictions = counts.argmax(dim=1)
    stable_distance = ((counts[:, 0] - counts[:, 1]).abs() / 2).ceil()  # vote flips required to change the prediction
    proposed_local_sens = 0.0
    global_sens = 1.0

    def private_prediction(eps: float) -> float:
        eps = eps / 2
        noise_distribution = torch.distributions.laplace.Laplace(0, 1 / eps + 1e-8)
        noise = noise_distribution.sample(stable_distance.shape).to(counts.device)
        vote_flips_private = stable_distance + noise
        test_result = vote_flips_private > np.log(1 / delta) / eps
        release_sensitivity = torch.where(
            test_result, proposed_local_sens * torch.ones_like(test_result), global_sens * torch.ones_like(test_result)
        )
        noise_distribution = torch.distributions.laplace.Laplace(0, release_sensitivity / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)
        assert noise.shape == predictions.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


def get_ensemble_stable_distance(
    bounded_model_ensemble: list[dict[int, BoundedModel]],
    batch: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """
    Get the maximum stable distance for each prediction of the ensemble
    """
    # compute the maximum stable distance for each prediction and each model
    max_ks = []
    for bounded_model_dict in bounded_model_ensemble:
        max_ks.append(privacy_utils.compute_max_certified_k(batch, bounded_model_dict).squeeze())
    max_ks = torch.stack(max_ks, dim=1)

    # compute the number of votes that have to flip to change the prediction
    # which is half the distance between the max and min votes
    vote_flips = torch.ceil((counts.max(dim=1).values - counts.min(dim=1).values) / 2)

    # sort the max_ks tensor, then for each row max_ks[i], sum the smallest vote_flips[i] values
    # this gives us a lower bound on the stable distance of the ensemble prediction
    max_ks = max_ks.sort(dim=1).values
    mask = torch.linspace(0, max_ks.shape[1] - 1, max_ks.shape[1], device=max_ks.device).view(1, max_ks.shape[1])
    mask = mask.repeat(max_ks.shape[0], 1) < vote_flips.view(max_ks.shape[0], 1)
    stable_distance = (max_ks * mask).sum(1)
    return stable_distance


def predict_ptr(
    bounded_model_ensemble: list[dict[int, BoundedModel]],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    delta: float,
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the ensemble on a batch of data with the PTR+AGT mechanism.
    """
    # get the model with k_private = 0 from the bounded_model_ensemble
    ensemble = [model_dict[0] for model_dict in bounded_model_ensemble]
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)
    predictions = counts.argmax(dim=1)
    proposed_local_sens = 0.0
    global_sens = 1.0

    stable_distance = get_ensemble_stable_distance(bounded_model_ensemble, batch, counts)

    def private_prediction(eps: float) -> float:
        eps = eps / 2
        noise_distribution = torch.distributions.laplace.Laplace(0, 1 / eps + 1e-8)
        noise = noise_distribution.sample(stable_distance.shape).to(counts.device)
        vote_flips_private = stable_distance + noise
        test_result = vote_flips_private > np.log(1 / delta) / eps
        release_sensitivity = torch.where(
            test_result, proposed_local_sens * torch.ones_like(test_result), global_sens * torch.ones_like(test_result)
        )
        noise_distribution = torch.distributions.laplace.Laplace(0, release_sensitivity / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)
        assert noise.shape == predictions.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


def predict_smooth_sens_cauchy(
    bounded_model_ensemble: list[dict[int, BoundedModel]],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the ensemble on a batch of data with the smooth sens and cauchy mechanism.
    """
    # get the model with k_private = 0 from the bounded_model_ensemble
    ensemble = [model_dict[0] for model_dict in bounded_model_ensemble]
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)
    predictions = counts.argmax(dim=1)

    stable_distance = get_ensemble_stable_distance(bounded_model_ensemble, batch, counts)

    def private_prediction(eps: float) -> float:
        beta = eps / 6
        smooth_sens_bound = torch.exp(-beta * stable_distance).reshape(predictions.shape)

        noise_distribution = torch.distributions.cauchy.Cauchy(0, 6 * smooth_sens_bound / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)
        assert predictions.shape == noise.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


def predict_smooth_sens_laplace(
    bounded_model_ensemble: list[dict[int, BoundedModel]],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    delta: float,
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the ensemble on a batch of data with the smooth sens and laplace mechanism.
    """
    # get the model with k_private = 0 from the bounded_model_ensemble
    ensemble = [model_dict[0] for model_dict in bounded_model_ensemble]
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)
    predictions = counts.argmax(dim=1)
    stable_distance = get_ensemble_stable_distance(bounded_model_ensemble, batch, counts)

    def private_prediction(eps: float) -> float:
        beta = eps / (2 * np.log(2 / delta))
        smooth_sens_bound = torch.exp(-beta * stable_distance).reshape(predictions.shape)

        noise_distribution = torch.distributions.laplace.Laplace(0, 2 * smooth_sens_bound / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)
        assert predictions.shape == noise.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


def predict_smooth_sens_cauchy_counts(
    bounded_model_ensemble: list[dict[int, BoundedModel]],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """Compute the accuracy of the ensemble using the smooth sensitivity of the counts."""
    # get the model with k_private = 0 from the bounded_model_ensemble
    ensemble = [model_dict[0] for model_dict in bounded_model_ensemble]
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)

    stable_distance = torch.inf * torch.ones(counts.size(0), device=counts.device)

    for model_dict in bounded_model_ensemble:
        k = privacy_utils.compute_max_certified_k(batch, model_dict).squeeze()
        stable_distance = torch.min(stable_distance, k)

    def private_prediction(eps: float) -> float:
        d = counts.size(1)
        beta = eps / (6 * d)
        smooth_sens_bound = 2.0 * torch.exp(-beta * stable_distance).reshape(labels.shape)
        noise_distribution = torch.distributions.cauchy.Cauchy(0, 6 * smooth_sens_bound / eps + 1e-8)
        noise = noise_distribution.sample((d,)).T.to(counts.device)  # type: ignore
        assert counts.shape == noise.shape
        private_predictions = (counts + noise).argmax(dim=1)
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


def predict_smooth_sens_cauchy_counts_2(
    bounded_model_ensemble: list[dict[int, BoundedModel]],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    *,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """Compute the accuracy of the ensemble using the smooth sensitivity of the counts."""
    # get the model with k_private = 0 from the bounded_model_ensemble
    ensemble = [model_dict[0] for model_dict in bounded_model_ensemble]
    counts, labels = get_ensemble_predictions(ensemble, batch, labels)

    stable_distance = torch.inf * torch.ones(counts.size(0), device=counts.device)

    for model_dict in bounded_model_ensemble:
        k = privacy_utils.compute_max_certified_k(batch, model_dict).squeeze()
        stable_distance = torch.min(stable_distance, k)

    def private_prediction(eps: float) -> float:
        assert counts.size(1) == 2
        beta = eps / 6
        smooth_sens_bound = 2.0 * torch.exp(-beta * stable_distance).reshape(labels.shape)
        noise_distribution = torch.distributions.cauchy.Cauchy(0, 6 * smooth_sens_bound / eps + 1e-8)
        noise = noise_distribution.sample().to(counts.device)  # type: ignore
        private_predictions = (counts[:, 1] - counts[:, 0] + noise) > 0
        return (private_predictions == labels).float().mean().item()

    return handle_repeats(private_prediction, epsilon, n_repeats)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import script_utils
    import training_utils
    import datasets
    import models

    config = training_utils.Config(
        learning_rate=2.0,
        l2_reg=0.02,
        n_epochs=2,
        device="cuda:1",
        loss="cross_entropy",
        lr_decay=0.6,
        lr_min=1e-3,
        clip_gamma=0.08,
        log_level="WARNING",
        seed=0,
        batchsize=5000,
    )

    ks_privacy = list(range(0, 200, 1))
    dataset_train_far, dataset_test_far = datasets.get_blobs(1.25, 0.35, 10000, 0)
    model = models.fully_connected(0)

    cache_name = f"blobs_far_{config.hash()}"

    T = 100

    ensemble_agt = training_utils.train_ensemble_agt(
        T,
        ks_privacy,
        models.fully_connected,
        config,
        dataset_train_far,
        cache_name,
    )

    ensemble = [model_dict[0] for model_dict in ensemble_agt]

    # %%

    epsilons = torch.logspace(-2, 2, 10)

    epsilon = 1.0
    delta = 10**-5

    test_batch, test_labels = dataset_test_far.tensors
    print(predict_noise_free(ensemble, test_batch, test_labels))
    # print(predict_global_sens(ensemble, test_batch, test_labels, epsilon))
    # print(predict_ptr_without_agt(ensemble, test_batch, test_labels, epsilon, delta))
    # print(predict_ptr(ensemble_agt, test_batch, test_labels, epsilon, delta))
    # print(predict_smooth_sens_cauchy(ensemble_agt, test_batch, test_labels, epsilon))
    # print(predict_smooth_sens_laplace(ensemble_agt, test_batch, test_labels, epsilon, delta))
    print(predict_smooth_sens_cauchy_counts(ensemble_agt, test_batch, test_labels, epsilon))
    print(predict_smooth_sens_cauchy_counts_2(ensemble_agt, test_batch, test_labels, epsilon))

    # %%

    def plot_mean_and_stds(acc_list, label, ax):
        means, stds = zip(*acc_list)
        ax.plot(epsilons, means, label=label)
        ax.fill_between(epsilons, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)

    n_repeats = 5
    epsilons = list(np.logspace(-2, 1, 20))
    epsilons = list(np.linspace(0.01, 0.5, 20))

    noise_free_acc = predict_noise_free(ensemble, test_batch, test_labels)

    fig, ax = plt.subplots(figsize=(5, 4))

    plot_mean_and_stds(
        predict_global_sens(ensemble, test_batch, test_labels, epsilons, n_repeats=n_repeats),
        "Global Sensitivity Laplace",
        ax,
    )
    plot_mean_and_stds(
        predict_ptr_without_agt(ensemble, test_batch, test_labels, epsilons, delta, n_repeats=n_repeats),
        "PTR no AGT",
        ax,
    )

    plot_mean_and_stds(
        predict_ptr(ensemble_agt, test_batch, test_labels, epsilons, delta, n_repeats=n_repeats), "PTR+AGT", ax
    )

    plot_mean_and_stds(
        predict_smooth_sens_cauchy(ensemble_agt, test_batch, test_labels, epsilons, n_repeats=n_repeats),
        "Smooth Sensitivity Cauchy",
        ax,
    )

    plot_mean_and_stds(
        predict_smooth_sens_cauchy_counts(ensemble_agt, test_batch, test_labels, epsilons, n_repeats=n_repeats),
        "Smooth Sensitivity Cauchy Counts",
        ax,
    )

    plot_mean_and_stds(
        predict_smooth_sens_cauchy_counts_2(ensemble_agt, test_batch, test_labels, epsilons, n_repeats=n_repeats),
        "Smooth Sensitivity Cauchy Counts 1d",
        ax,
    )

    plot_mean_and_stds(
        predict_smooth_sens_laplace(ensemble_agt, test_batch, test_labels, epsilons, delta, n_repeats=n_repeats),
        "Smooth Sensitivity Laplace",
        ax,
    )

    ax.axhline(noise_free_acc, color="black", linestyle="--", label="Noise-free")
    ax.legend()
    # ax.set_xscale("log")
    ax.set_xlabel("$\epsilon$ per Query ")
    ax.set_ylabel("Accuracy")

    script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.8), dpi=300)
