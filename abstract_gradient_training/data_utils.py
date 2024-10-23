"""
Helper functions for data loading in AGT. Dataloading in AGT has a few additional requirements compared to standard
dataloader usage in PyTorch. This includes handling incomplete batches and pairs of dataloaders, e.g. one for clean data
and one for poisoned data (or public / private data).
"""

import logging
from collections.abc import Iterator, Iterable

import torch

LOGGER = logging.getLogger(__name__)


def dataloader_wrapper(dl_train: Iterable, n_epochs: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Return a new generator that iterates over the training dataloader for a fixed number of epochs.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    We assume the first batch is full to set the batchsize and this is compared with all subsequent batches.

    Args:
        - dl_train (Iterable): Training dataloader that returns (batch, labels) tuples at each iteration.
        - n_epochs (int): Number of epochs to iterate over the dataloader.

    Yields:
        -batch, labels: Post-processed (batch, labels) tuples.
    """
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epoch
    for n in range(n_epochs):
        LOGGER.info("Starting epoch %s", n + 1)
        t = -1  # possibly undefined loop variable
        for t, (batch, labels) in enumerate(dl_train):
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batch.size(0)
                LOGGER.debug("Initialising dataloader batchsize to %s", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batch.size(0) != full_batchsize:
                LOGGER.debug(
                    "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                    t + 1,
                    n + 1,
                    full_batchsize,
                    batch.size(0),
                )
                continue
            # return the batches for this iteration
            yield batch, labels
        # check the number of batches we have processed and report the appropriate warnings
        assert t != -1, f"Dataloader is empty at epoch {n + 1}!"
        if t == 1:
            LOGGER.warning("Dataloader has only one batch per epoch, effective batchsize may be smaller than expected.")


def dataloader_pair_wrapper(
    dl_train: Iterable, dl_aux: Iterable | None, n_epochs: int, dtype: torch.dtype
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    """
    Return a new generator that iterates over the training dataloaders for a fixed number of epochs.
    The first dataloader contains the standard training data, while the second dataloader contains auxiliary data,
    which is e.g. clean data for poisoning or public data for privacy.
    For each combined batch, we return one batch from the clean dataloader and one batch from the poisoned dataloader.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    We assume the first batch is full to set the batchsize and this is compared with all subsequent batches.

    Args:
        dl_train (Iterable): Dataloader that returns (batch, labels) tuples at each iteration.
        dl_aux (Iterable | None): Optional additional dataloader for auxiliary data that returns (batch, labels)
            tuples at each iteration.
        n_epochs (int): Maximum number of epochs.
        dtype (torch.dtype): Datatype to convert the batch tensors to.

    Yields:
        batch, labels, batch_aux, labels_aux: Tuples of post-processed (batch, labels, batch_aux, labels_aux)
            for each iteration.
    """
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epochs
    for n in range(n_epochs):
        LOGGER.info("Starting epoch %s", n + 1)
        # handle the case where there is no auxiliary dataloader by returning dummy values
        if dl_aux is None:
            data_iterator: Iterable = (((b, l), (None, None)) for b, l in dl_train)
        else:
            data_iterator = zip(dl_train, dl_aux)  # note that zip will stop at the shortest iterator
        t = -1  # possibly undefined loop variable
        for t, ((batch, labels), (batch_aux, labels_aux)) in enumerate(data_iterator):
            batch = batch.to(dtype)
            batchsize = batch.size(0)
            if batch_aux is not None:
                batchsize += batch_aux.size(0)
                batch_aux = batch_aux.to(dtype)
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batchsize
                LOGGER.debug("Initialising dataloader batchsize to %s", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batchsize != full_batchsize:
                LOGGER.debug(
                    "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                    t + 1,
                    n + 1,
                    full_batchsize,
                    batchsize,
                )
                continue
            # return the batches for this iteration
            yield batch, labels, batch_aux, labels_aux
        # check the number of batches we have processed and report the appropriate warnings
        assert t != -1, f"Dataloader is empty at epoch {n + 1}!"
        if t == 1:
            LOGGER.warning("Dataloader has only one batch per epoch, effective batchsize may be smaller than expected.")
