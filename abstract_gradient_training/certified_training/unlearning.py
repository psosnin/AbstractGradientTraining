import math
import torch
from tqdm import trange

from abstract_gradient_training import nominal_pass
from abstract_gradient_training import definitions as agt_definitions
from abstract_gradient_training.certified_training import utils as ct_utils
from abstract_gradient_training import bound_utils
from abstract_gradient_training import test_metrics


@torch.no_grad()
def unlearning_certified_training(model, config, dl, dl_test, transform=None):
    """
    Train the neural network and get certified unlearning bounds.
    Parameters:
        model: neural network model
        config: configuration dictionary
        dl: dataloader for the training set
        dl_test: dataloader for the test set
        transform: optional transform to apply to the input data (e.g. propagating through fixed conv layers)
    """
    config = ct_utils.validate_config(config)
    device = torch.device(config["device"])
    model = model.to(dl.dataset[0][0].dtype).to(device)  # match the dtype and device of the model and data
    param_n, param_l, param_u = ct_utils.get_parameters(model)
    batchsize = config["batchsize"]
    n_epochs = config["n_epochs"]
    fragsize = config["fragsize"]
    optimizer = agt_definitions.OPTIMIZERS[config["optimizer"]](param_n, config)
    loss = config["loss"]
    k_unlearn = config["k_unlearn"]
    gamma = config["clip_gamma"]
    sigma = config["dp_sgd_sigma"]
    noise_level = 0.0 if math.isnan(sigma * gamma) else sigma * gamma  # to account for gamma = inf
    sound = (noise_level == 0.0)

    # start the training loop
    progress_bar = trange(n_epochs)
    for _ in progress_bar:
        # loop over all batches
        for t, (batch, labels) in enumerate(dl):
            if batch.size(0) != batchsize:
                assert t != 0, "First batch must be full"
                continue
            network_eval = test_metrics.evaluate_network(param_n, param_l, param_u, dl_test, loss, model, transform)
            if ct_utils.break_condition(network_eval):
                return param_l, param_n, param_u, network_eval

            # we want the shape to be [batchsize x input_dim x 1]
            if transform is None:
                batch = batch.view(batch.size(0), -1, 1).type(param_n[-1].dtype)

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
                progress_bar.set_postfix_str(
                    f"eval: {network_eval} bound: {(param_l[0] - param_u[0]).norm():.3} batch: {t} frag: {f}"
                )
                batch_frag = batch_fragments[f].to(device)
                label_frag = label_fragments[f].to(device)
                batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
                logit_n, inter_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
                _, _, dL_n = agt_definitions.LOSS_BOUNDS[loss](logit_n, logit_n, logit_n, label_frag)
                frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, inter_n)
                frag_grads_l, frag_grads_u = ct_utils.grads_helper(batch_frag, batch_frag, label_frag, param_l, param_u, config)

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
                    bound_utils.validate_interval(grads_l[i], grads_n[i])
                    bound_utils.validate_interval(grads_n[i], grads_u[i])
                else:
                    bound_utils.validate_interval(grads_l[i], grads_u[i])
                grads_n[i] += torch.normal(torch.zeros_like(grads_n[i]), noise_level)

            param_n, param_l, param_u = optimizer.step(
                param_n, param_l, param_u, grads_n, grads_l, grads_u, sound=sound
            )

    return param_l, param_n, param_u, network_eval
