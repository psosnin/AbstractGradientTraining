import logging
import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.certified_training.configuration import AGTConfig

"""
Helper functions for certified training.
"""

def grads_helper(
    batch_l: torch.Tensor,
    batch_u: torch.Tensor,
    labels: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    config: AGTConfig,
    label_poison: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Helper function to calculate bounds on the gradient of the loss function with respect to all parameters given the
    input and parameter bounds.
    
    Args:
        batch_l (torch.Tensor): [fragsize x input_dim x 1] tensor of inputs to the network.
        batch_u (torch.Tensor): [fragsize x input_dim x 1] tensor of inputs to the network.
        labels (torch.Tensor): [fragsize, ] tensor of labels for the inputs.
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        config (AGTConfig): Configuration object for the abstract gradient training module.
        label_poison (bool, optional): Boolean flag to indicate if the labels are being poisoned. Defaults to False.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: _description_
    """
    labels = labels.squeeze()
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # get config parameters
    bound_kwargs = config.bound_kwargs
    loss_bound_fn = config.loss_bound_fn
    forward_bound_fn = config.forward_bound_fn
    backward_bound_fn = config.backward_bound_fn
    label_epsilon = config.label_epsilon if label_poison else 0.0
    k_label_poison = config.label_k_poison if label_poison else 0
    poison_target = config.poison_target if label_poison else -1
    # forward pass through the network with bounds
    logit_l, logit_u, inter_l, inter_u = forward_bound_fn(
        param_l, param_u, batch_l, batch_u, **bound_kwargs
    )
    # calculate the first partial derivative of the loss function
    # (pass logit_u in as a dummy for logit_n and ignore dL_n)
    dL_l, dL_u, _ = loss_bound_fn(
        logit_l, logit_u, logit_u, labels, k_label_poison=k_label_poison,
        label_epsilon=label_epsilon, poison_target=poison_target
    )
    # compute backwards pass through the network with bounds
    grad_min, grad_max = backward_bound_fn(
        dL_l, dL_u, param_l, param_u, inter_l, inter_u, **bound_kwargs
    )

    return grad_min, grad_max


def break_condition(evaluation):
    """
    Check whether to terminate the certified training loop based on the bounds on the test metric (MSE or Accuracy).
    eval[0] = worst case eval
    eval[1] = nominal eval
    eval[2] = best case eval
    """
    if evaluation[0] <= 0.03 and evaluation[2] >= 0.97:  # worst case accuracy bounds too loose
        logging.warning("Early stopping due to loose bounds")
        return True
    if evaluation[0] >= 1e2:  # worst case MSE too large
        logging.warning("Early stopping due to loose bounds")
        return True


def get_parameters(model):
    """
    Get the parameters of the dense layers of the pytorch model.
    """
    param_n = [(l.weight, l.bias) for l in model.modules() if isinstance(l, torch.nn.Linear)]  # get linear params
    param_n = [item for sublist in param_n for item in sublist]  # flatten the list
    param_n = [t if len(t.shape) == 2 else t.unsqueeze(-1) for t in param_n]  # reshape bias to [n x 1] instead of [n]
    param_n = [t.detach().clone() for t in param_n]
    param_l = [p.clone() for p in param_n]
    param_u = [p.clone() for p in param_n]
    return param_n, param_l, param_u


def propagate_conv_layers(x, model, epsilon):
    """
    Propagate an input batch through the convolutional layers of a model. Here we assume that the conv layers are all
    at the start of the network with ReLU activations after each one.
    """
    # get the parameters of the conv layers
    conv_layers = [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]
    conv_parameters = [(l.weight.detach(), l.bias.detach(), l.stride, l.padding) for l in conv_layers]
    # propagate the input through the conv layers
    x_l, x_u = x - epsilon, x + epsilon
    for W, b, stride, padding in conv_parameters:
        x_l, x_u = interval_arithmetic.propagate_conv2d(x_l, x_u, W, b, stride, padding)
        x_l, x_u = torch.nn.functional.relu(x_l), torch.nn.functional.relu(x_u)
    x_l = x_l.flatten(start_dim=1)
    x_u = x_u.flatten(start_dim=1)
    return x_l.unsqueeze(-1), x_u.unsqueeze(-1)


def dataloader_wrapper(dl_train, n_epochs):
    """
    Return a new generator that iterates over the training dataloader for a fixed number of epochs.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    Note that we assume the first batch is full.
    """
    if len(dl_train) == 1:
        logging.warning("Dataloader has only one batch, effective batchsize may be smaller than expected.")
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epoch
    for n in range(n_epochs):
        for t, (batch, labels) in enumerate(dl_train):
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batch.size(0)
                logging.debug("Initialising dataloader batchsize to {}", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batch.size(0) != full_batchsize:
                logging.debug("Skipping incomplete batch {} in epoch {}", t, n)
                continue
            # return the batches for this iteration
            yield batch, labels
            

def dataloader_pair_wrapper(dl_train, dl_clean, n_epochs):
    """
    Return a new generator that iterates over the training dataloaders for a fixed number of epochs.
    For each combined batch, we return one batch from the clean dataloader and one batch from the poisoned dataloader.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    Note that we assume the first batch is full.
    """
    # this is the max number of batches, if none are skipped due to being incomplete
    max_batches_per_epoch = len(dl_train) if dl_clean is None else min(len(dl_train), len(dl_clean))
    if max_batches_per_epoch == 1:
        logging.warning("Dataloader has only one batch, effective batchsize may be smaller than expected.")
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epochs
    for n in range(n_epochs):
        # handle the case where there is no clean dataloader by returning dummy values
        if dl_clean is None:
            data_iterator = (((b, l), (None, None)) for b, l in dl_train)
        else:
            # note that zip will stop at the shortest iterator
            data_iterator = zip(dl_train, dl_clean)
        for t, ((batch, labels), (batch_clean, labels_clean)) in enumerate(data_iterator):
            # check the length of this batch
            batch_len = batch.size(0)
            if batch_clean is not None:
                batch_len += batch_clean.size(0)
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batch.size(0)
                logging.debug("Initialising dataloader batchsize to {}", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batch.size(0) != full_batchsize:
                logging.debug(
                    "Skipping incomplete batch {} in epoch {} (expected batchsize {}, got {})",
                    t, n, full_batchsize, batch_len
                )
                continue
            # return the batches for this iteration
            yield batch, labels, batch_clean, labels_clean