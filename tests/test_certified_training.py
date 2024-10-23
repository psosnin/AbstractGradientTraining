from itertools import product
from parameterized import parameterized
from copy import deepcopy
import torch

from tests import utils
from abstract_gradient_training import poison_certified_training
from abstract_gradient_training import privacy_certified_training
from abstract_gradient_training import unlearning_certified_training


@parameterized.expand(
    product(
        utils.EPSILONS,
        utils.K_POISON,
        utils.BATCHSIZES,
        range(utils.N_SEEDS),
    )
)
def test_poisoning(epsilon, k_poison, batchsize, seed):
    """
    Test certified poisoning training on a uci regression dataset. We first generate a nominal network and then
    train it using the certified training module.
    """
    torch.manual_seed(seed)
    # initialize the model and dataset
    dl_train, dl_test = utils.get_dataloaders(batchsize, batchsize)
    model = utils.FullyConnected(12, 1, 64, 1)  # network with 1 hidden layer of 64 neurons
    # feature poisoning
    config = deepcopy(utils.NOMINAL_CONFIG)
    config.k_poison = k_poison
    config.epsilon = epsilon
    param_l, param_n, param_u = poison_certified_training(model, config, dl_train, dl_test)
    utils.validate_sound(param_l, param_n)
    utils.validate_sound(param_n, param_u)


@parameterized.expand(
    product(
        utils.K_PRIVATE,
        utils.BATCHSIZES,
        range(utils.N_SEEDS),
    )
)
def test_privacy(k_private, batchsize, seed):
    """
    Test certified privacy training on a uci regression dataset. We first generate a nominal network and then
    train it using the certified training module.
    """
    torch.manual_seed(seed)
    # initialize the model and dataset
    dl_train, dl_test = utils.get_dataloaders(batchsize, batchsize)
    model = utils.FullyConnected(12, 1, 64, 1)  # network with 1 hidden layer of 64 neurons
    config = deepcopy(utils.NOMINAL_CONFIG)
    config.k_private = k_private
    param_l, param_n, param_u = privacy_certified_training(model, config, dl_train, dl_test)
    utils.validate_sound(param_l, param_n)
    utils.validate_sound(param_n, param_u)


@parameterized.expand(
    product(
        utils.K_UNLEARN,
        utils.BATCHSIZES,
        range(utils.N_SEEDS),
    )
)
def test_unlearning(k_unlearn, batchsize, seed):
    """
    Test certified unlearning training on a uci regression dataset. We first generate a nominal network and then
    train it using the certified training module.
    """
    torch.manual_seed(seed)
    # initialize the model and dataset
    dl_train, dl_test = utils.get_dataloaders(batchsize, batchsize)
    model = utils.FullyConnected(12, 1, 64, 1)  # network with 1 hidden layer of 64 neurons
    config = deepcopy(utils.NOMINAL_CONFIG)
    config.k_unlearn = k_unlearn
    param_l, param_n, param_u = unlearning_certified_training(model, config, dl_train, dl_test)
    utils.validate_sound(param_l, param_n)
    utils.validate_sound(param_n, param_u)
