"""Poison certified training."""

from __future__ import annotations
import logging
from collections.abc import Iterable

import torch

from abstract_gradient_training import training_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training.bounded_models import BoundedModel

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def poison_certified_training(
    bounded_model: BoundedModel,
    config: AGTConfig,
    dl_train: Iterable,
    dl_val: Iterable | None = None,
    dl_clean: Iterable | None = None,
) -> BoundedModel:
    """
    Train the model with the given config and return a model with poison-certified bounds on the parameters.

    Args:
        bounded_model (BoundedModel): Bounded version of the pytorch model to train.
        config (ct_config.AGTConfig): Configuration object (see agt.certified_training.configuration.py for details)
        dl_train (Iterable): Iterable for the training data that returns (batch, labels) tuples at each iteration.
        dl_val (Iterable): Iterable for the validation data that returns (batch, labels) tuples at each iteration. This
            data is used to log the performance of the model at each training iteration.
        dl_clean (Iterable, optional): Iterable for "clean" training data. If provided, a batch will be taken from
            both dl_train and dl_clean for each training batch. Poisoned bounds will only be calculated for the batches
            from dl_train.

    Returns:
        BoundedModel: The trained model with the final bounds on the parameters.
    """

    # initialise hyperparameters, model, data, optimizer, logging
    bounded_model.to(config.device)
    optimizer = config.get_bounded_optimizer(bounded_model)
    training_utils.log_run_start(config, agt_type="poison")

    if config.paired_poison:
        k_poison = max(config.k_poison, config.label_k_poison)
    else:
        k_poison = config.k_poison + config.label_k_poison

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = training_utils.dataloader_pair_wrapper(dl_train, dl_clean, config.n_epochs)
    val_iterator = training_utils.dataloader_cycle(dl_val) if dl_val is not None else None

    for n, (batch, labels, batch_clean, labels_clean) in enumerate(training_iterator, 1):
        config.on_iter_start_callback(bounded_model)
        # possibly terminate early
        if config.early_stopping_callback(bounded_model):
            break

        # evaluate the network on the validation data and log the result
        if val_iterator is not None:
            loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
            LOGGER.info(f"Batch {n}. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in bounded_model.param_n]  # nominal gradients
        grads_u = [torch.zeros_like(p) for p in bounded_model.param_n]  # upper bound gradients
        grads_l = [torch.zeros_like(p) for p in bounded_model.param_n]  # lower bound gradients
        grads_diffs_l = [[] for _ in bounded_model.param_n]  # difference of input+weight and weight perturbed bounds
        grads_diffs_u = [[] for _ in bounded_model.param_n]  # difference of input+weight and weight perturbed bounds

        # process clean data
        batch_fragments = torch.split(batch_clean, config.fragsize, dim=0) if batch_clean is not None else []
        label_fragments = torch.split(labels_clean, config.fragsize, dim=0) if labels_clean is not None else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # nominal pass
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True, poisoned=False
            )
            # weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False, poisoned=False
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
            # nominal pass
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True, poisoned=False
            )
            # weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False, poisoned=False
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
            # input + weight perturbed bounds
            frag_grads_iwp_l, frag_grads_iwp_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False, poisoned=True
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

        # normalise each by the batchsize and apply the step with the optimizer
        batchsize = batch.size(0) if batch_clean is None else batch.size(0) + batch_clean.size(0)
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]
        interval_arithmetic.validate_interval(grads_l, grads_u, grads_n, msg=f"grad bounds, batch {n}")
        optimizer.step(grads_l, grads_n, grads_u)
        config.on_iter_end_callback(bounded_model)

    if val_iterator is not None:
        loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
        LOGGER.info(f"Final Eval. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

    training_utils.validate_bounded_model(bounded_model)
    LOGGER.info("=================== Finished Poison Certified Training ===================")

    return bounded_model
