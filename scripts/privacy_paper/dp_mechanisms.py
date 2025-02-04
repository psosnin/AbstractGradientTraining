"""
Differentially private mechanisms for the privacy paper.

Since we are using these only for generating plots, we allow each prediction mechanism to
take in a list of epsilons and also perform multiple runs so that we can average the noisy accuracies.
Performing predictions like this is not safe for real-world privacy guarantees, its just faster to re-use intermediate
results for generating plots.
"""

# %%

from collections.abc import Callable
import numpy as np
import torch
from abstract_gradient_training import bounded_models, privacy_utils

# Private prediction for single models


def get_model_predictions(
    model: torch.nn.Sequential | bounded_models.BoundedModel, batch: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the predictions of a model on a batch of data.
    """
    if isinstance(model, bounded_models.BoundedModel):
        device = torch.device(model.device) if model.device != -1 else torch.device("cpu")
    else:
        device = torch.device(model.parameters().__next__().device)

    batch, labels = batch.to(device), labels.to(device)

    logits = model.forward(batch)

    if logits.size(1) == 1:
        predictions = torch.round(torch.sigmoid(logits))
    elif logits.size(1) == 2:
        predictions = torch.argmax(logits, dim=1)
    else:
        raise ValueError("Only binary classification is supported.")

    return predictions.reshape(labels.shape), labels


def handle_epsilon_list_and_repeats(
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
    model: torch.nn.Sequential | bounded_models.BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute the accuracy of the model on a batch of data.
    """
    predictions, labels = get_model_predictions(model, batch, labels)
    return (predictions == labels).float().mean().item()


def predict_global_sens(
    model: torch.nn.Sequential | bounded_models.BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the model on a batch of data with the Laplace mechanism and the global
    sensitivity of the prediction.

    The mechanism is (epsilon, 0)-differentially private.

    """
    predictions, labels = get_model_predictions(model, batch, labels)

    def private_prediction(eps: float) -> float:
        noise_distribution = torch.distributions.laplace.Laplace(0, 1 / eps + 1e-8)
        noise = noise_distribution.sample(predictions.shape).to(predictions.device)
        assert predictions.shape == noise.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_epsilon_list_and_repeats(private_prediction, epsilon, n_repeats)


def predict_smooth_sens_laplace(
    bounded_model_dict: dict[int, bounded_models.BoundedModel],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    delta: float,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the model on a batch of data with the Laplace mechanism and the smooth sensitivity.
    """
    # handle devices
    model = bounded_model_dict[0]
    predictions, labels = get_model_predictions(model, batch, labels)

    # compute the max stable distance
    max_k = privacy_utils.compute_max_certified_k(batch, bounded_model_dict)
    print("Smooth Sens Laplace, max_k:", max_k.mean())

    def private_prediction(eps: float) -> float:
        beta = eps / (2 * np.log(2 / delta))
        smooth_sens_bound = torch.exp(-beta * max_k).reshape(labels.shape)

        noise_distribution = torch.distributions.laplace.Laplace(0, 2 * smooth_sens_bound / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)
        assert predictions.shape == noise.shape
        private_predictions = (predictions + noise) > 0.5

        return (private_predictions == labels).float().mean().item()

    return handle_epsilon_list_and_repeats(private_prediction, epsilon, n_repeats)


def predict_smooth_sens_cauchy(
    bounded_model_dict: dict[int, bounded_models.BoundedModel],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the model on a batch of data with the Cauchy mechanism and the smooth sensitivity.
    """
    model = bounded_model_dict[0]
    predictions, labels = get_model_predictions(model, batch, labels)

    # compute the max certified distance
    max_k = privacy_utils.compute_max_certified_k(batch, bounded_model_dict)

    def private_prediction(eps: float) -> float:
        beta = eps / 6
        smooth_sens_bound = torch.exp(-beta * max_k).reshape(labels.shape)

        noise_distribution = torch.distributions.cauchy.Cauchy(0, 6 * smooth_sens_bound / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)
        assert predictions.shape == noise.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_epsilon_list_and_repeats(private_prediction, epsilon, n_repeats)


def predict_ptr(
    bounded_model_dict: dict[int, bounded_models.BoundedModel],
    batch: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float | list[float],
    delta: float,
    n_repeats: int = 1,
) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Compute the accuracy of the model on a batch of data with the PTR mechanism.
    """
    model = bounded_model_dict[0]
    predictions, labels = get_model_predictions(model, batch, labels)

    # compute the max stable distance
    max_k = privacy_utils.compute_max_certified_k(batch, bounded_model_dict).reshape(labels.shape)

    # propose
    proposed_local_sensitivity = 0.0
    global_sens = 1.0

    def private_prediction(eps: float) -> float:
        # epsilon is halved for each step
        eps = eps / 2

        noise_distribution = torch.distributions.laplace.Laplace(0, 1 / eps + 1e-8)
        noise = noise_distribution.sample(max_k.shape).to(predictions.device)
        assert noise.shape == max_k.shape
        max_k_private = max_k + noise
        test_result = max_k_private > np.log(1 / delta) / eps
        # where the test passes, use the proposed local sensitivity
        # where the test fails, use the global sensitivity
        release_sensitivity = torch.where(
            test_result,
            torch.ones_like(test_result) * proposed_local_sensitivity,
            torch.ones_like(test_result) * global_sens,
        )

        # release
        noise_distribution = torch.distributions.laplace.Laplace(0, release_sensitivity / eps + 1e-8)
        noise = noise_distribution.sample().to(predictions.device)

        # add noise the predictions
        assert predictions.shape == noise.shape
        private_predictions = (predictions + noise) > 0.5
        return (private_predictions == labels).float().mean().item()

    return handle_epsilon_list_and_repeats(private_prediction, epsilon, n_repeats)


# %%
if __name__ == "__main__":
    import script_utils
    import blobs_train

    ks_privacy = list(range(0, 200, 1))
    dataset_train_far, dataset_test_far = blobs_train.get_dataset(1.25, 0.35)
    model = blobs_train.get_model()
    model = blobs_train.train_model(model, dataset_train_far)

    bounded_model_dict = blobs_train.run_sweep(ks_privacy, dataset_train_far, dataset_test_far, True)

    test_point, test_label = dataset_test_far.tensors

    # %%

    epsilon = 2.0
    delta = 10**-5

    print(predict_noise_free(model, test_point, test_label))
    print(predict_global_sens(model, test_point, test_label, epsilon))
    print(predict_smooth_sens_laplace(bounded_model_dict, test_point, test_label, epsilon, delta))
    print(predict_smooth_sens_cauchy(bounded_model_dict, test_point, test_label, epsilon))
    print(predict_ptr(bounded_model_dict, test_point, test_label, epsilon, delta))

    # %%

    import matplotlib.pyplot as plt

    n_repeats = 10

    epsilons = list(np.logspace(-2, 1, 20))

    noise_free_acc = predict_noise_free(model, test_point, test_label)

    global_sens_accs = predict_global_sens(model, test_point, test_label, epsilons, n_repeats)

    smooth_sens_laplace_accs = predict_smooth_sens_laplace(
        bounded_model_dict, test_point, test_label, epsilons, delta, n_repeats
    )
    smooth_sens_cauchy_accs = predict_smooth_sens_cauchy(
        bounded_model_dict, test_point, test_label, epsilons, n_repeats
    )
    ptr_accs = predict_ptr(bounded_model_dict, test_point, test_label, epsilons, delta, n_repeats)

    fig, ax = plt.subplots(figsize=(5, 4))

    def plot_mean_and_stds(acc_list, label, ax):
        means, stds = zip(*acc_list)
        ax.plot(epsilons, means, label=label)
        ax.fill_between(epsilons, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)

    plot_mean_and_stds(global_sens_accs, "Global Sensitivity Laplace", ax)
    plot_mean_and_stds(smooth_sens_laplace_accs, "Smooth Sensitivity Laplace", ax)
    plot_mean_and_stds(smooth_sens_cauchy_accs, "Smooth Sensitivity Cauchy", ax)
    plot_mean_and_stds(ptr_accs, "PTR", ax)
    ax.axhline(noise_free_acc, color="black", linestyle="--", label="Noise-free")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("$\epsilon$ per Query ")
    ax.set_ylabel("Accuracy")

    script_utils.apply_figure_size(fig, script_utils.set_size(1.0, shrink_height=0.8), dpi=300)
