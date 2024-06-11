import torch
from tqdm import trange

from abstract_gradient_training import nominal_pass
from abstract_gradient_training import definitions as agt_definitions
from abstract_gradient_training.certified_training import utils as ct_utils
from abstract_gradient_training import bound_utils
from abstract_gradient_training import test_metrics

@torch.no_grad()
def poison_certified_training(model, config, dl, dl_test, dl_clean=None, transform=None):
    """
    Train the neural network while tracking lower and upper bounds on all the parameters under a possible poisoning
    attack.
    For each training batch, take a batch from both dl and dl_clean.
    Parameters:
        model: neural network model
        config: configuration dictionary
        dl: dataloader for the training data
        dl_test: dataloader for the test data
        dl_clean: dataloader for additional provably clean training data
        transform: optional transform to apply to the input data that depends on the model (e.g. fixed conv layers)
    """
    # initialise the training parameterse
    config = ct_utils.validate_config(config)
    device = torch.device(config["device"])
    dtype = dl.dataset[0][0].dtype
    model = model.to(dtype).to(device)  # match the dtype and device of the model and data
    param_n, param_l, param_u = ct_utils.get_parameters(model)
    epsilon = config["epsilon"]
    k_poison = max(config["k_poison"], config["label_k_poison"])
    batchsize = config["batchsize"]
    n_epochs = config["n_epochs"]
    fragsize = config["fragsize"]
    optimizer = agt_definitions.OPTIMIZERS[config["optimizer"]](param_n, config)
    loss = config["loss"]

    # start the certified training loop
    prog_bar = trange(n_epochs)
    for _ in prog_bar:
        train_iter = iter(dl)
        train_iter_clean = iter(dl_clean) if dl_clean else None
        for t in range(len(dl)):
            # get the potentially poisoned and clean data
            batch, labels = next(train_iter)
            batch_clean, labels_clean = next(train_iter_clean) if dl_clean else (None, None)
            batch_len = batch.size(0) + batch_clean.size(0) if dl_clean else batch.size(0)

            # check the batch is the correct size
            if batch_len != batchsize:
                assert t != 0, "First batch must be full"
                continue

            # evaluate the network, log and possibly terminage early
            network_eval = test_metrics.evaluate_network(param_n, param_l, param_u, dl_test, loss, model, transform)
            prog_bar.set_postfix_str(f"eval: {network_eval} bound: {(param_l[0] - param_u[0]).norm():.3} batch: {t}")
            if ct_utils.break_condition(network_eval):
                return param_l, param_n, param_u, network_eval

            # we want the shape to be [batchsize x input_dim x 1]
            if transform is None:
                batch = batch.view(batch.size(0), -1, 1).type(dtype)
                batch_clean = batch_clean.view(batch_clean.size(0), -1, 1).type(dtype) if dl_clean else None

            # initialise containers to store the nominal and bounds on the gradients for each fragment
            # the bounds are stored as lists of lists indexed by [parameter, fragment]
            grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
            grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradients
            grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradients
            grads_diffs_l = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds
            grads_diffs_u = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds

            # process clean data
            batch_fragments = torch.split(batch_clean, fragsize, dim=0) if dl_clean else []
            label_fragments = torch.split(labels_clean, fragsize, dim=0) if dl_clean else []
            for batch_frag, label_frag in zip(batch_fragments, label_fragments):
                batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
                batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
                # nominal pass
                logit_n, inter_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
                _, _, dL_n = agt_definitions.LOSS_BOUNDS[loss](logit_n, logit_n, logit_n, label_frag)
                frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, inter_n)
                # weight perturbed bounds
                grads_weight_perturb_l, grads_weight_perturb_u = ct_utils.grads_helper(
                    batch_frag, batch_frag, label_frag, param_l, param_u, config, False
                )
                grads_n = [a + b.sum(dim=0) for a, b in zip(grads_n, frag_grads_n)]
                grads_l = [a + b.sum(dim=0) for a, b in zip(grads_l, grads_weight_perturb_l)]
                grads_u = [a + b.sum(dim=0) for a, b in zip(grads_u, grads_weight_perturb_u)]

            # process potentially poisoned data
            batch_fragments = torch.split(batch, fragsize, dim=0)
            label_fragments = torch.split(labels, fragsize, dim=0)
            for batch_frag, label_frag in zip(batch_fragments, label_fragments):
                batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
                # nominal pass
                batch_frag_n = transform(batch_frag, model, 0)[0] if transform else batch_frag
                logit_n, inter_n = nominal_pass.nominal_forward_pass(batch_frag_n, param_n)
                _, _, dL_n = agt_definitions.LOSS_BOUNDS[loss](logit_n, logit_n, logit_n, label_frag)
                frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, inter_n)
                # weight perturbed bounds
                grads_weight_perturb_l, grads_weight_perturb_u = ct_utils.grads_helper(
                    batch_frag_n, batch_frag_n, label_frag, param_l, param_u, config, False
                )
                grads_n = [a + b.sum(dim=0) for a, b in zip(grads_n, frag_grads_n)]
                grads_l = [a + b.sum(dim=0) for a, b in zip(grads_l, grads_weight_perturb_l)]
                grads_u = [a + b.sum(dim=0) for a, b in zip(grads_u, grads_weight_perturb_u)]
                # apply input transformation
                if transform:
                    batch_frag_l, batch_frag_u = transform(batch_frag, model, epsilon)
                else:
                    batch_frag_l, batch_frag_u = batch_frag - epsilon, batch_frag + epsilon
                # input + weight perturbed bounds
                grads_input_weight_perturb_l, grads_input_weight_perturb_u = ct_utils.grads_helper(
                    batch_frag_l, batch_frag_u, label_frag, param_l, param_u, config, True
                )
                # calculate differences between the input+weight perturbed and weight perturbed bounds
                diffs_l = [a - b for a, b in zip(grads_input_weight_perturb_l, grads_weight_perturb_l)]  # -ve
                diffs_l = [torch.topk(a, k_poison, dim=0, largest=False)[0] for a in diffs_l]
                [g.append(d) for g, d in zip(grads_diffs_l, diffs_l)]
                diffs_u = [a - b for a, b in zip(grads_input_weight_perturb_u, grads_weight_perturb_u)]  # +ve
                diffs_u = [torch.topk(a, k_poison, dim=0)[0] for a in diffs_u]
                [g.append(d) for g, d in zip(grads_diffs_u, diffs_u)]

            # accumulate the top-k diffs from each fragment then add the overall top-k diffs to the gradient bounds
            grads_diffs_l = [torch.cat(g, dim=0) for g in grads_diffs_l]
            grads_diffs_u = [torch.cat(g, dim=0) for g in grads_diffs_u]
            for i in range(len(grads_n)):
                grads_l[i] += torch.topk(grads_diffs_l[i], k_poison, dim=0, largest=False)[0].sum(dim=0)
                grads_u[i] += torch.topk(grads_diffs_u[i], k_poison, dim=0)[0].sum(dim=0)
                bound_utils.validate_interval(grads_l[i], grads_n[i])
                bound_utils.validate_interval(grads_n[i], grads_u[i])

            # normalise each by the batchsize
            grads_l = [g / batchsize for g in grads_l]
            grads_n = [g / batchsize for g in grads_n]
            grads_u = [g / batchsize for g in grads_u]

            param_n, param_l, param_u = optimizer.step(param_n, param_l, param_u, grads_n, grads_l, grads_u)

    return param_l, param_n, param_u, network_eval
