"""Poison certified training."""

from __future__ import annotations
import itertools
import logging
from collections.abc import Callable, Iterable

import torch

from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training import training_utils
from abstract_gradient_training import model_utils
from abstract_gradient_training import data_utils
from abstract_gradient_training import nominal_pass
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training import optimizers

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def poison_certified_training(
    model: torch.nn.Sequential,
    config: AGTConfig,
    dl_train: Iterable,
    dl_test: Iterable,
    dl_clean: Iterable | None = None,
    transform: Callable | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Train the neural network while tracking lower and upper bounds on all the parameters under a possible poisoning
    attack.

    Args:
        model (torch.nn.Sequential): Neural network model to train. Expected to be a Sequential model with linear
            layers and ReLU activations on all layers except the last.
        config (ct_config.AGTConfig): Configuration object (see agt.certified_training.configuration.py for details)
        dl_train (Iterable): Iterable for the training data that returns (batch, labels) tuples at each iteration.
        dl_test (Iterable): Iterable for the testing data that returns (batch, labels) tuples at each iteration.
        dl_clean (Iterable, optional): Iterable for "clean" training data. If provided, a batch will be taken from
            both dl_train and dl_clean for each training batch. Poisoned bounds will only be calculated for the batches
            from dl_train.
        transform (Callable | None): Optional transform to apply to the data before passing through the model. The
            transform function takes an input batch and an epsilon value and returns lower and upper bounds of the
            transformed input.

    Returns:
        param_l: list of lower bounds on the parameters of the linear layers of the model [W1, b1, ..., Wm, bm]
        param_n: list of nominal values of the parameters of the linear layers of the model [W1, b1, ..., Wm, bm]
        param_u: list of upper bounds on the parameters of the linear layers of the model [W1, b1, ..., Wm, bm]
    """

    # initialise hyperparameters, model, data, optimizer, logging
    device = torch.device(config.device)
    model = model.to(device)  # match the device of the model and data
    param_l, param_n, param_u = model_utils.get_parameters(model)
    k_poison = max(config.k_poison, config.label_k_poison)
    optimizer = optimizers.SGD(config)

    # set up logging and print run info
    logging.getLogger("abstract_gradient_training").setLevel(config.log_level)
    LOGGER.info("=================== Starting Poison Certified Training ===================")
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
    LOGGER.debug("\tAdversary feature-space budget: epsilon=%s, k_poison=%s", config.epsilon, config.k_poison)
    LOGGER.debug(
        "\tAdversary label-space budget: label_epsilon=%s, label_k_poison=%s, poison_target=%s",
        config.label_epsilon,
        config.label_k_poison,
        config.poison_target,
    )
    LOGGER.debug("\tClipping: gamma=%s, method=%s", config.clip_gamma, config.clip_method)
    LOGGER.debug(
        "\tBounding methods: forward=%s, loss=%s, backward=%s", config.forward_bound, config.loss, config.backward_bound
    )

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = data_utils.dataloader_pair_wrapper(dl_train, dl_clean, config.n_epochs, param_n[-1].dtype)
    test_iterator = itertools.cycle(dl_test)

    for n, (batch, labels, batch_clean, labels_clean) in enumerate(training_iterator):
        # evaluate the network
        network_eval = config.test_loss_fn(param_l, param_n, param_u, *next(test_iterator), transform=transform)

        # possibly terminate early
        if config.early_stopping and training_utils.break_condition(network_eval):
            break
        config.callback(network_eval, param_l, param_n, param_u)

        # log the current network evaluation
        LOGGER.info("Training batch %s: %s", n + 1, training_utils.get_progress_message(network_eval, param_l, param_u))

        # calculate batchsize
        batchsize = batch.size(0) if batch_clean is None else batch.size(0) + batch_clean.size(0)

        # we want the shape to be [batchsize x input_dim x 1]
        if transform is None:
            batch = batch.view(batch.size(0), -1, 1)
            batch_clean = batch_clean.view(batch_clean.size(0), -1, 1) if batch_clean is not None else None

        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradients
        grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradients
        grads_diffs_l = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds
        grads_diffs_u = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds

        # process clean data
        batch_fragments = torch.split(batch_clean, config.fragsize, dim=0) if batch_clean is not None else []
        label_fragments = torch.split(labels_clean, config.fragsize, dim=0) if labels_clean is not None else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
            batch_frag = transform(batch_frag, 0)[0] if transform else batch_frag
            # nominal pass
            activations_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            _, _, dl_n = config.loss_bound_fn(activations_n[-1], activations_n[-1], activations_n[-1], label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dl_n, param_n, activations_n)
            # weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = training_utils.grads_helper(
                batch_frag, batch_frag, label_frag, param_l, param_u, config, False
            )
            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i] = training_utils.propagate_clipping(
                    frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i], config.clip_gamma, config.clip_method
                )
                # accumulate the gradients
                grads_l[i] = grads_l[i] + frag_grads_wp_l[i].sum(dim=0)
                grads_n[i] = grads_n[i] + frag_grads_n[i].sum(dim=0)
                grads_u[i] = grads_u[i] + frag_grads_wp_u[i].sum(dim=0)

        # process potentially poisoned data
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
            # nominal pass
            batch_frag_n = transform(batch_frag, 0)[0] if transform else batch_frag
            activations_n = nominal_pass.nominal_forward_pass(batch_frag_n, param_n)
            _, _, dl_n = config.loss_bound_fn(activations_n[-1], activations_n[-1], activations_n[-1], label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dl_n, param_n, activations_n)
            # weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = training_utils.grads_helper(
                batch_frag_n, batch_frag_n, label_frag, param_l, param_u, config, False
            )
            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i] = training_utils.propagate_clipping(
                    frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i], config.clip_gamma, config.clip_method
                )
                # accumulate the gradients
                grads_l[i] = grads_l[i] + frag_grads_wp_l[i].sum(dim=0)
                grads_n[i] = grads_n[i] + frag_grads_n[i].sum(dim=0)
                grads_u[i] = grads_u[i] + frag_grads_wp_u[i].sum(dim=0)

            # apply input transformation
            if transform:
                batch_frag_l, batch_frag_u = transform(batch_frag, config.epsilon)
            else:
                batch_frag_l, batch_frag_u = batch_frag - config.epsilon, batch_frag + config.epsilon
            # input + weight perturbed bounds
            frag_grads_iwp_l, frag_grads_iwp_u = training_utils.grads_helper(
                batch_frag_l, batch_frag_u, label_frag, param_l, param_u, config, True
            )
            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_iwp_l[i], _, frag_grads_iwp_u[i] = training_utils.propagate_clipping(
                    frag_grads_iwp_l[i], torch.zeros(1), frag_grads_iwp_u[i], config.clip_gamma, config.clip_method
                )
                # calculate the differences beetween the input+weight perturbed and weight perturbed bounds
                diffs_l = frag_grads_iwp_l[i] - frag_grads_wp_l[i]
                diffs_u = frag_grads_iwp_u[i] - frag_grads_wp_u[i]
                # accumulate and store the the top-k diffs from each fragment
                grads_diffs_l[i].append(torch.topk(diffs_l, k_poison, dim=0, largest=False)[0])
                grads_diffs_u[i].append(torch.topk(diffs_u, k_poison, dim=0)[0])

        # accumulate the top-k diffs from each fragment then add the overall top-k diffs to the gradient bounds
        for i in range(len(grads_n)):
            # we pop, process and del each one by one to save memory
            grads_diffs_l_i = grads_diffs_l.pop(0)
            grads_diffs_l_i = torch.cat(grads_diffs_l_i, dim=0)
            grads_l[i] += torch.topk(grads_diffs_l_i, k_poison, dim=0, largest=False)[0].sum(dim=0)
            del grads_diffs_l_i
            grads_diffs_u_i = grads_diffs_u.pop(0)
            grads_diffs_u_i = torch.cat(grads_diffs_u_i, dim=0)
            grads_u[i] += torch.topk(grads_diffs_u_i, k_poison, dim=0)[0].sum(dim=0)
            del grads_diffs_u_i
            interval_arithmetic.validate_interval(grads_l[i], grads_u[i], grads_n[i])

        # normalise each by the batchsize
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]

        param_l, param_n, param_u = optimizer.step(param_l, param_n, param_u, grads_n, grads_l, grads_u)

    network_eval = config.test_loss_fn(param_l, param_n, param_u, *next(test_iterator), transform=transform)
    LOGGER.info("Final network eval: %s", training_utils.get_progress_message(network_eval, param_l, param_u))

    LOGGER.info("=================== Finished Poison Certified Training ===================")

    return param_l, param_n, param_u
