"""Poison certified training."""

from __future__ import annotations
from collections.abc import Iterable
import logging
import gc

import torch

from abstract_gradient_training import training_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.gradient_accumulation import PoisoningGradientAccumulator
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

    # initialise the gradient accumulation class, which handles the logic of accumulating gradients across batch
    # fragments and computing the certified descent direction bounds.
    gradient_accumulator = PoisoningGradientAccumulator(k_poison, bounded_model.param_n)

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = training_utils.dataloader_pair_wrapper(dl_train, dl_clean, config.n_epochs)
    val_iterator = training_utils.dataloader_cycle(dl_val) if dl_val is not None else None

    # main training loop
    for n, (batch, labels, batch_clean, labels_clean) in enumerate(training_iterator, 1):
        config.on_iter_start_callback(bounded_model)
        # possibly terminate early
        if config.early_stopping_callback(bounded_model):
            break

        # evaluate the network on the validation data and log the result
        if val_iterator is not None:
            loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
            LOGGER.info(f"Batch {n}. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

        # NOTE: We have a lot of per-sample operations that must be performed on the entire batchsize, with typically
        # very large batchsizes. To conserve memory, we split the batch into fragments and process each fragment
        # separately and accumulate the gradients across fragments using the gradient_accumulator classes.

        # process clean data
        batch_fragments = torch.split(batch_clean, config.fragsize, dim=0) if batch_clean is not None else []
        label_fragments = torch.split(labels_clean, config.fragsize, dim=0) if labels_clean is not None else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # compute nominal pass
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True, poisoned=False
            )
            # compute weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False, poisoned=False
            )
            # clip the gradients
            frag_grads_wp_l, frag_grads_n, frag_grads_wp_u = training_utils.propagate_clipping(
                frag_grads_wp_l, frag_grads_n, frag_grads_wp_u, config.clip_gamma, config.clip_method
            )
            # accumulate the gradients
            gradient_accumulator.add_clean_fragment_gradients(frag_grads_n, frag_grads_wp_l, frag_grads_wp_u)
            # the gpu memory allocations at each loop are not always collected, so we'll prompt pytorch to do so
            gc.collect()
            torch.cuda.empty_cache()

        # process potentially poisoned data
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # compute nominal gradients
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True, poisoned=False
            )
            # compute weight perturbed gradients
            frag_grads_wp_l, frag_grads_wp_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False, poisoned=False
            )
            # compute the input+weight perturbed gradients
            frag_grads_iwp_l, frag_grads_iwp_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False, poisoned=True
            )
            # apply gradient clipping
            frag_grads_wp_l, frag_grads_n, frag_grads_wp_u = training_utils.propagate_clipping(
                frag_grads_wp_l, frag_grads_n, frag_grads_wp_u, config.clip_gamma, config.clip_method
            )
            frag_grads_iwp_l, _, frag_grads_iwp_u = training_utils.propagate_clipping(
                frag_grads_iwp_l, frag_grads_n, frag_grads_iwp_u, config.clip_gamma, config.clip_method
            )
            # accumulate the gradients
            gradient_accumulator.add_poisoned_fragment_gradients(
                frag_grads_n, frag_grads_wp_l, frag_grads_wp_u, frag_grads_iwp_l, frag_grads_iwp_u
            )
            # the gpu memory allocations at each loop are not always collected, so we'll prompt pytorch to do so
            gc.collect()
            torch.cuda.empty_cache()

        # get the bounds on the descent direction, validate them, and apply the optimizer update
        batchsize = batch.size(0) if batch_clean is None else batch.size(0) + batch_clean.size(0)
        update_l, update_n, update_u = gradient_accumulator.concretize_gradient_update(batchsize)
        interval_arithmetic.validate_interval(update_l, update_u, update_n, msg=f"grad bounds, batch {n}")
        optimizer.step(update_l, update_n, update_u)
        config.on_iter_end_callback(bounded_model)

    if val_iterator is not None:
        loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
        LOGGER.info(f"Final Eval. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

    training_utils.validate_bounded_model(bounded_model)
    LOGGER.info("=================== Finished Poison Certified Training ===================")

    return bounded_model
