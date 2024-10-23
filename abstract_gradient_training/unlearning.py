"""Certified unlearning training."""

from __future__ import annotations
import logging
import itertools
from collections.abc import Callable, Iterable

import torch

from abstract_gradient_training import nominal_pass
from abstract_gradient_training import training_utils
from abstract_gradient_training import model_utils
from abstract_gradient_training import data_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training import optimizers


LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def unlearning_certified_training(
    model: torch.nn.Sequential,
    config: AGTConfig,
    dl_train: Iterable,
    dl_test: Iterable,
    transform: Callable | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Train the dense layers of a neural network with the given config and return the unlearning-certified bounds
    on the parameters.

    Args:
        model (torch.nn.Sequential): Neural network model. Must be a torch.nn.Sequential object with dense layers and
            ReLU activations only.
        config (AGTConfig): Configuration object for the abstract gradient training module. See the configuration module
            for more details.
        dl_train (Iterable): Iterable of training data that returns (batch, labels) tuples at each iteration.
        dl_test (Iterable): Iterable of testing data that returns (batch, labels) tuples at each iteration.
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.

    Returns:
        param_l (list[torch.Tensor]): List of lower bounds of the trained parameters [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of nominal trained parameters [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of upper bounds of the trained parameters [W1, b1, ..., Wn, bn].
    """
    # set up logging
    logging.getLogger("abstract_gradient_training").setLevel(config.log_level)
    LOGGER.info("=================== Starting Unlearning Certified Training ===================")
    LOGGER.debug(
        "\tOptimizer params: n_epochs=%s, learning_rate=%s, l1_reg=%s, l2_reg=%s",
        config.n_epochs,
        config.learning_rate,
        config.l1_reg,
        config.l2_reg,
    )
    LOGGER.debug(
        "\tLearning rate schedule: lr_decay=%s, lr_min=%s, early_stopping=%s",
        config.lr_decay,
        config.lr_min,
        config.early_stopping,
    )
    LOGGER.debug("\tUnlearning parameter: k_unlearn=%s", config.k_unlearn)
    LOGGER.debug("\tClipping: gamma=%s, method=%s", config.clip_gamma, config.clip_method)
    LOGGER.debug("\tNoise: type=%s, multiplier=%s", config.noise_type, config.noise_multiplier)
    LOGGER.debug(
        "\tBounding methods: forward=%s, loss=%s, backward=%s", config.forward_bound, config.loss, config.backward_bound
    )

    # initialise hyperparameters, model, data, optimizer, logging
    device = torch.device(config.device)
    model = model.to(device)  # match the device of the model and data
    param_l, param_n, param_u = model_utils.get_parameters(model)
    optimizer = optimizers.SGD(config)
    k_unlearn = config.k_unlearn
    noise_distribution = config.noise_distribution

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = data_utils.dataloader_wrapper(dl_train, config.n_epochs)
    test_iterator = itertools.cycle(dl_test)

    for n, (batch, labels) in enumerate(training_iterator):
        # evaluate the network
        network_eval = config.test_loss_fn(param_l, param_n, param_u, *next(test_iterator), transform=transform)
        # decide whether to terminate training early
        if config.early_stopping and training_utils.break_condition(network_eval):
            break
        config.callback(network_eval, param_l, param_n, param_u)
        # log the current network evaluation
        LOGGER.info("Training batch %s: %s", n + 1, training_utils.get_progress_message(network_eval, param_l, param_u))
        # we want the shape to be [batchsize x input_dim x 1]
        if transform is None:
            batch = batch.view(batch.size(0), -1, 1).type(param_n[-1].dtype)
        batchsize = batch.size(0)
        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradient
        grads_l = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        grads_l_top_ks = [[] for _ in param_n]  # top k lower bound gradients from each fragment
        grads_u_bottom_ks = [[] for _ in param_n]  # bottom k upper bound gradients from each fragment

        # split the batch into fragments to avoid running out of GPU memory
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for f in range(len(batch_fragments)):  # loop over all batch fragments
            batch_frag = batch_fragments[f].to(device)
            label_frag = label_fragments[f].to(device)
            batch_frag = transform(batch_frag, 0)[0] if transform else batch_frag
            activations_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            logit_n = activations_n[-1]
            _, _, dl_n = config.loss_bound_fn(logit_n, logit_n, logit_n, label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dl_n, param_n, activations_n)
            frag_grads_l, frag_grads_u = training_utils.grads_helper(
                batch_frag, batch_frag, label_frag, param_l, param_u, config
            )

            # accumulate the results for this batch to save memory
            for i in range(len(grads_n)):
                # clip the gradients
                frag_grads_l[i], frag_grads_n[i], frag_grads_u[i] = training_utils.propagate_clipping(
                    frag_grads_l[i], frag_grads_n[i], frag_grads_u[i], config.clip_gamma, config.clip_method
                )
                # accumulate the nominal gradients
                grads_n[i] += frag_grads_n[i].sum(dim=0)
                # accumulate the top/bottom s - k gradient bounds
                size = frag_grads_n[i].size(0)
                # we are guaranteed to take the bottom s - k from the lower bound, so add the sum to grads_l
                # the remaining k gradients are stored until all the frags have been processed
                top_k_l = torch.topk(frag_grads_l[i], min(size, k_unlearn), largest=True, dim=0)[0]
                grads_l[i] += frag_grads_l[i].sum(dim=0) - top_k_l.sum(dim=0)
                grads_l_top_ks[i].append(top_k_l)
                # we are guaranteed to take the top s - k from the upper bound, so add the sum to grads_u
                # the remaining k gradients are stored until all the frags have been processed
                bottom_k_u = torch.topk(frag_grads_u[i], min(size, k_unlearn), largest=False, dim=0)[0]
                grads_u[i] += frag_grads_u[i].sum(dim=0) - bottom_k_u.sum(dim=0)
                grads_u_bottom_ks[i].append(bottom_k_u)

        # Apply the unlearning update mechanism to the bounds.
        for i in range(len(grads_n)):
            # do these separately to conserve memory
            grads_l_top_ks_i = grads_l_top_ks.pop(0)
            grads_l_top_ks_i = torch.cat(grads_l_top_ks_i, dim=0)
            assert grads_l_top_ks_i.size(0) >= k_unlearn, "Not enough samples left after processing batch fragments."
            top_k_l = torch.topk(grads_l_top_ks_i, k_unlearn, largest=True, dim=0)[0]
            grads_l[i] += grads_l_top_ks_i.sum(dim=0) - top_k_l.sum(dim=0)
            del grads_l_top_ks_i

            grads_u_bottom_ks_i = grads_u_bottom_ks.pop(0)
            grads_u_bottom_ks_i = torch.cat(grads_u_bottom_ks_i, dim=0)
            bottom_k_u = torch.topk(grads_u_bottom_ks_i, k_unlearn, largest=False, dim=0)[0]
            grads_u[i] += grads_u_bottom_ks_i.sum(dim=0) - bottom_k_u.sum(dim=0)
            del grads_u_bottom_ks_i

        # normalise each by the batchsize
        grads_l = [g / (batchsize - k_unlearn) for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / (batchsize - k_unlearn) for g in grads_u]

        # check bounds and add noise
        for i in range(len(grads_n)):
            interval_arithmetic.validate_interval(grads_l[i], grads_u[i], grads_n[i])
            noise = noise_distribution(grads_n[i].size()).to(device)
            grads_l[i] += noise
            grads_n[i] += noise
            grads_u[i] += noise

        param_l, param_n, param_u = optimizer.step(param_l, param_n, param_u, grads_n, grads_l, grads_u)

    network_eval = config.test_loss_fn(param_l, param_n, param_u, *next(test_iterator), transform=transform)
    LOGGER.info("Final network eval: %s", training_utils.get_progress_message(network_eval, param_l, param_u))

    for i in range(len(param_n)):
        violations = (param_l[i] > param_n[i]).sum() + (param_n[i] > param_u[i]).sum()
        max_violation = max((param_l[i] - param_n[i]).max().item(), (param_n[i] - param_u[i]).max().item())
        if violations > 0:
            LOGGER.warning("Nominal parameters not within certified bounds for parameter %s", i)
            LOGGER.debug("\tNumber of violations: %s", violations.item())
            LOGGER.debug("\tMax violation: %.2e", max_violation)

    LOGGER.info("=================== Finished Unlearning Certified Training ===================")

    return param_l, param_n, param_u
