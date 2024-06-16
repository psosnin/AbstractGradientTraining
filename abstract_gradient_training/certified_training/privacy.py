import logging
import torch
from tqdm import tqdm

from abstract_gradient_training import nominal_pass
from abstract_gradient_training.certified_training import utils as ct_utils
from abstract_gradient_training import bound_utils
from abstract_gradient_training import test_metrics


@torch.no_grad()
def privacy_certified_training(model, config, dl_train, dl_test, transform=None):
    """
    Train the neural network using dp-sgd and certified privacy.
    NOTE: The returned nominal parameters are not guaranteed to be inside the parameter bounds if dp_sgd is used.
    Parameters:
        model: neural network model
        config: configuration dictionary
        dl_train: dataloader for the training set
        dl_test: dataloader for the test set
        transform: optional transform to apply to the input data (e.g. propagating through fixed conv layers)
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
    gamma = config.clip_gamma
    sigma = config.dp_sgd_sigma
    k_private = config.k_private

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = ct_utils.dataloader_wrapper(dl_train, n_epochs)
    training_iterator = tqdm(training_iterator, desc="Training Batch", disable=quiet)

    for batch, labels in training_iterator:
        # evaluate the network and log the results to tqdm
        network_eval = test_loss_fn(param_n, param_l, param_u, dl_test, model, transform)
        training_iterator.set_postfix_str(
            f"Network eval: Worst={network_eval[0]:.2g}, Nominal={network_eval[1]:.2g}, Best={network_eval[2]:.2g}"
        )
        # get if we should terminate training early
        if ct_utils.break_condition(network_eval):
            return param_l, param_n, param_u, network_eval
        # we want the shape to be [batchsize x input_dim x 1]
        if transform is None:
            batch = batch.view(batch.size(0), -1, 1).type(param_n[-1].dtype)
        batchsize = batch.size(0)
        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        grads_l = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        grads_l_top_ks = [[] for _ in param_n]  # top k lower bound gradients from each fragment
        grads_u_bottom_ks = [[] for _ in param_n]  # bottom k upper bound gradients from each fragment

        # split the batch into fragments to avoid running out of GPU memory
        batch_fragments = torch.split(batch, fragsize, dim=0)
        label_fragments = torch.split(labels, fragsize, dim=0)
        for f in range(len(batch_fragments)):  # loop over all batch fragments
            batch_frag = batch_fragments[f].to(device)
            batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
            label_frag = label_fragments[f].to(device)
            logit_n, inter_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            _, _, dL_n = loss_bound_fn(logit_n, logit_n, logit_n, label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, inter_n)
            frag_grads_l, frag_grads_u = ct_utils.grads_helper(batch_frag, batch_frag, label_frag, param_l, param_u, config)

            # clip the gradients
            frag_grads_l = [torch.clamp(g, -gamma, gamma) for g in frag_grads_l]
            frag_grads_u = [torch.clamp(g, -gamma, gamma) for g in frag_grads_u]
            frag_grads_n = [torch.clamp(g, -gamma, gamma) for g in frag_grads_n]

            # accumulate the results for this batch to save memory
            grads_n = [a + b.sum(dim=0) for a, b in zip(grads_n, frag_grads_n)]
            for i in range(len(grads_n)):
                size = frag_grads_n[i].size(0)
                # we are guaranteed to take the bottom s - k from the lower bound, so add the sum to grads_l
                # the remaining k gradients are stored until all the frags have been processed
                top_k_l = torch.topk(frag_grads_l[i], min(size, k_private), largest=True, dim=0)[0]
                grads_l[i] += frag_grads_l[i].sum(dim=0) - top_k_l.sum(dim=0)
                grads_l_top_ks[i].append(top_k_l)

                # we are guaranteed to take the top s - k from the upper bound, so add the sum to grads_u
                # the remaining k gradients are stored until all the frags have been processed
                bottom_k_u = torch.topk(frag_grads_u[i], min(size, k_private), largest=False, dim=0)[0]
                grads_u[i] += frag_grads_u[i].sum(dim=0) - bottom_k_u.sum(dim=0)
                grads_u_bottom_ks[i].append(bottom_k_u)

        # concatenate
        grads_l_top_ks = [torch.cat(g, dim=0) for g in grads_l_top_ks]
        grads_u_bottom_ks = [torch.cat(g, dim=0) for g in grads_u_bottom_ks]

        # Apply the unlearning update mechanism to the bounds.
        for i in range(len(grads_n)):
            size = grads_l_top_ks[i].size(0)
            assert size >= k_private, "Not enough samples left after processing batch fragments."
            top_k_l = torch.topk(grads_l_top_ks[i], k_private, largest=True, dim=0)[0]
            bottom_k_u = torch.topk(grads_u_bottom_ks[i], k_private, largest=False, dim=0)[0]
            grads_l[i] += grads_l_top_ks[i].sum(dim=0) - top_k_l.sum(dim=0) - k_private * gamma
            grads_u[i] += grads_u_bottom_ks[i].sum(dim=0) - bottom_k_u.sum(dim=0) + k_private * gamma

        # normalise each by the batchsize
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]

        # check bounds and add noise
        for i in range(len(grads_n)):
            if sigma == 0.0:  # sound update
                bound_utils.validate_interval(grads_l[i], grads_n[i])
                bound_utils.validate_interval(grads_n[i], grads_u[i])
            else:  # unsound update due to noise
                bound_utils.validate_interval(grads_l[i], grads_u[i])
            grads_n[i] += torch.normal(torch.zeros_like(grads_n[i]), sigma * gamma)

        param_n, param_l, param_u = optimizer.step(
            param_n, param_l, param_u, grads_n, grads_l, grads_u, sound=False
        )

    return param_l, param_n, param_u