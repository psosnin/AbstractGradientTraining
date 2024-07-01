"""Certified unlearning training."""

from __future__ import annotations
from typing import Optional, Callable
import logging
import math

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from abstract_gradient_training import nominal_pass
from abstract_gradient_training.certified_training import utils as ct_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.certified_training.configuration import AGTConfig


@torch.no_grad()
def unlearning_certified_training(
    model: torch.nn.Sequential,
    config: AGTConfig,
    dl_train: DataLoader,
    dl_test: DataLoader,
    transform: Optional[Callable] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Train the dense layers of a neural network with the given config and return the unlearning-certified bounds
    on the parameters.

    Args:
        model (torch.nn.Sequential): Neural network model. Must be a torch.nn.Sequential object with dense layers and
                                     ReLU activations only. The model may have other layers (e.g. convolutional layers)
                                     before the dense section, but these must be fixed and are not trained. If fixed
                                     non-dense layers are provided, then the transform function must be set to propagate
                                     bounds through these layers.
        config (AGTConfig): Configuration object for the abstract gradient training module. See the configuration module
                            for more details.
        dl_train (DataLoader): Training data loader.
        dl_test (DataLoader): Testing data loader.
        transform (Optional[Callable], optional): Optional function to propagate bounds through fixed layers of the
                                                  neural network (e.g. convolutional layers). Defaults to None.

    Returns:
        param_l (list[torch.Tensor]): List of lower bounds of the trained parameters [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of nominal trained parameters [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of upper bounds of the trained parameters [W1, b1, ..., Wn, bn].
    """

    # initialise hyperparameters, model, data, optimizer, logging
    device = torch.device(config.device)
    dtype = dl_train.dataset[0][0].dtype
    model = model.to(dtype).to(device)  # match the dtype and device of the model and data
    param_n, param_l, param_u = ct_utils.get_parameters(model)
    n_epochs = config.n_epochs
    fragsize = config.fragsize
    optimizer = config.optimizer_class(param_n, config)
    loss_bound_fn = config.loss_bound_fn
    test_loss_fn = config.test_loss_fn
    logging.getLogger().setLevel(config.log_level)
    quiet = logging.getLogger().getEffectiveLevel() > logging.INFO
    k_unlearn = config.k_unlearn
    gamma = config.clip_gamma
    sigma = config.dp_sgd_sigma
    noise_level = 0.0 if math.isnan(sigma * gamma) else sigma * gamma  # to account for gamma = inf
    sound = noise_level == 0.0

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = ct_utils.dataloader_wrapper(dl_train, n_epochs)
    training_iterator = tqdm(training_iterator, desc="Training Batch", disable=quiet)

    for batch, labels in training_iterator:
        # evaluate the network and log the results to tqdm
        network_eval = test_loss_fn(param_n, param_l, param_u, dl_test, model, transform)
        training_iterator.set_postfix_str(ct_utils.get_progress_message(network_eval, param_l, param_u))
        # decide whether to terminate training early
        if ct_utils.break_condition(network_eval):
            return param_l, param_n, param_u
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
        batch_fragments = torch.split(batch, fragsize, dim=0)
        label_fragments = torch.split(labels, fragsize, dim=0)
        for f in range(len(batch_fragments)):  # loop over all batch fragments
            batch_frag = batch_fragments[f].to(device)
            label_frag = label_fragments[f].to(device)
            batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
            activations_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            logit_n = activations_n[-1]
            _, _, dL_n = loss_bound_fn(logit_n, logit_n, logit_n, label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, activations_n)
            frag_grads_l, frag_grads_u = ct_utils.grads_helper(
                batch_frag, batch_frag, label_frag, param_l, param_u, config
            )

            # clip the gradients
            frag_grads_l = [torch.clamp(g, -gamma, gamma) for g in frag_grads_l]
            frag_grads_u = [torch.clamp(g, -gamma, gamma) for g in frag_grads_u]
            frag_grads_n = [torch.clamp(g, -gamma, gamma) for g in frag_grads_n]

            grads_n = [a + b.sum(dim=0) for a, b in zip(grads_n, frag_grads_n)]

            # accumulate the results for this batch to save memory
            for i in range(len(grads_n)):
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

        # concatenate
        grads_l_top_ks = [torch.cat(g, dim=0) for g in grads_l_top_ks]
        grads_u_bottom_ks = [torch.cat(g, dim=0) for g in grads_u_bottom_ks]

        # Apply the unlearning update mechanism to the bounds.
        for i in range(len(grads_n)):
            size = grads_l_top_ks[i].size(0)
            assert size >= k_unlearn, "Not enough samples left after processing batch fragments."
            top_k_l = torch.topk(grads_l_top_ks[i], k_unlearn, largest=True, dim=0)[0]
            bottom_k_u = torch.topk(grads_u_bottom_ks[i], k_unlearn, largest=False, dim=0)[0]
            grads_l[i] += grads_l_top_ks[i].sum(dim=0) - top_k_l.sum(dim=0)
            grads_u[i] += grads_u_bottom_ks[i].sum(dim=0) - bottom_k_u.sum(dim=0)

        # normalise each by the batchsize
        grads_l = [g / (batchsize - k_unlearn) for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / (batchsize - k_unlearn) for g in grads_u]

        # check bounds and add noise
        for i in range(len(grads_n)):
            if sound:
                interval_arithmetic.validate_interval(grads_l[i], grads_n[i])
                interval_arithmetic.validate_interval(grads_n[i], grads_u[i])
            else:
                interval_arithmetic.validate_interval(grads_l[i], grads_u[i])
            grads_n[i] += torch.normal(torch.zeros_like(grads_n[i]), noise_level)

        param_n, param_l, param_u = optimizer.step(param_n, param_l, param_u, grads_n, grads_l, grads_u, sound=sound)

    return param_l, param_n, param_u
