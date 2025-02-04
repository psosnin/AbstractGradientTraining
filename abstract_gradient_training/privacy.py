"""Certified privacy training."""

from __future__ import annotations
from collections.abc import Iterable
import logging

import torch

from abstract_gradient_training import training_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training.bounded_models import BoundedModel

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def privacy_certified_training(
    bounded_model: BoundedModel,
    config: AGTConfig,
    dl_train: Iterable,
    dl_val: Iterable | None = None,
    dl_public: Iterable | None = None,
) -> BoundedModel:
    """
    Train the model with the given config and return a model with privacy-certified bounds on the parameters.

    Args:
        bounded_model (BoundedModel): Bounded version of the pytorch model to train.
        config (AGTConfig): Configuration object for the abstract gradient training module. See the configuration module
            for more details.
        dl_train (Iterable): Iterable of training data that returns (batch, labels) tuples at each iteration.
        dl_val (Iterable): Iterable for the validation data that returns (batch, labels) tuples at each iteration. This
            data is used to log the performance of the model at each training iteration.
        dl_public (Iterable, optional): Iterable of data considered 'public' for the purposes of privacy certification.

    Returns:
        BoundedModel: The trained model with the final bounds on the parameters.
    """

    # initialise hyperparameters, model, data, optimizer, logging
    bounded_model.to(config.device)
    optimizer = config.get_bounded_optimizer(bounded_model)
    noise_distribution = config.get_noise_distribution()
    training_utils.log_run_start(config, agt_type="privacy")

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = training_utils.dataloader_pair_wrapper(dl_train, dl_public, config.n_epochs)
    val_iterator = training_utils.dataloader_cycle(dl_val) if dl_val is not None else None

    for n, (batch, labels, batch_public, labels_public) in enumerate(training_iterator, 1):
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
        grads_l = [torch.zeros_like(p) for p in bounded_model.param_n]  # upper bound gradient
        grads_u = [torch.zeros_like(p) for p in bounded_model.param_n]  # upper bound gradient
        grads_l_top_ks = [[] for _ in bounded_model.param_n]  # top k lower bound gradients from each fragment
        grads_u_bottom_ks = [[] for _ in bounded_model.param_n]  # bottom k upper bound gradients from each fragment

        # process the "public" data first
        batch_fragments = torch.split(batch_public, config.fragsize, dim=0) if batch_public is not None else []
        label_fragments = torch.split(labels_public, config.fragsize, dim=0) if labels_public is not None else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # nominal pass
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True
            )
            # weight perturbed bounds
            frag_grads_l, frag_grads_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False
            )

            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_l[i], frag_grads_n[i], frag_grads_u[i] = training_utils.propagate_clipping(
                    frag_grads_l[i], frag_grads_n[i], frag_grads_u[i], config.clip_gamma, config.clip_method
                )
                # accumulate the gradients
                grads_l[i] = grads_l[i] + frag_grads_l[i].sum(dim=0)
                grads_n[i] = grads_n[i] + frag_grads_n[i].sum(dim=0)
                grads_u[i] = grads_u[i] + frag_grads_u[i].sum(dim=0)

        # process the "private" data, taking the appropriate bounds
        # split the batch into fragments to avoid running out of GPU memory
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # nominal pass
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True
            )
            # weight perturbed bounds
            frag_grads_l, frag_grads_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False
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
                top_k_l = torch.topk(frag_grads_l[i], min(size, config.k_private), largest=True, dim=0)[0]
                grads_l[i] += frag_grads_l[i].sum(dim=0) - top_k_l.sum(dim=0)
                grads_l_top_ks[i].append(top_k_l)
                # we are guaranteed to take the top s - k from the upper bound, so add the sum to grads_u
                # the remaining k gradients are stored until all the frags have been processed
                bottom_k_u = torch.topk(frag_grads_u[i], min(size, config.k_private), largest=False, dim=0)[0]
                grads_u[i] += frag_grads_u[i].sum(dim=0) - bottom_k_u.sum(dim=0)
                grads_u_bottom_ks[i].append(bottom_k_u)

        # Apply the unlearning update mechanism to the bounds.
        for i in range(len(grads_n)):
            # do these separately to conserve memory
            grads_l_top_ks_i = grads_l_top_ks.pop(0)
            grads_l_top_ks_i = torch.cat(grads_l_top_ks_i, dim=0)
            assert grads_l_top_ks_i.size(0) >= config.k_private, "Not enough samples left after processing fragments."
            top_k_l = torch.topk(grads_l_top_ks_i, config.k_private, largest=True, dim=0)[0]
            grads_l[i] += grads_l_top_ks_i.sum(dim=0) - top_k_l.sum(dim=0) - config.k_private * config.clip_gamma
            del grads_l_top_ks_i

            grads_u_bottom_ks_i = grads_u_bottom_ks.pop(0)
            grads_u_bottom_ks_i = torch.cat(grads_u_bottom_ks_i, dim=0)
            bottom_k_u = torch.topk(grads_u_bottom_ks_i, config.k_private, largest=False, dim=0)[0]
            grads_u[i] += grads_u_bottom_ks_i.sum(dim=0) - bottom_k_u.sum(dim=0) + config.k_private * config.clip_gamma
            del grads_u_bottom_ks_i

        # normalise each by the batchsize
        batchsize = batch.size(0) if batch_public is None else batch.size(0) + batch_public.size(0)
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]

        # check bounds and add noise
        for i in range(len(grads_n)):
            interval_arithmetic.validate_interval(grads_l[i], grads_u[i], grads_n[i])
            noise = noise_distribution(grads_n[i].size())
            grads_l[i] += noise
            grads_n[i] += noise
            grads_u[i] += noise

        config.on_iter_end_callback(bounded_model)
        optimizer.step(grads_l, grads_n, grads_u)

    if val_iterator is not None:
        loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
        LOGGER.info(f"Final Eval. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

    training_utils.validate_bounded_model(bounded_model)
    LOGGER.info("=================== Finished Privacy Certified Training ===================")

    return bounded_model
