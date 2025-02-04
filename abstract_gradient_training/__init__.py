"""
Abstract gradient training (AGT) is a framework for training neural networks with certified robustness guarantees.
"""

import logging
from abstract_gradient_training.poisoning import poison_certified_training
from abstract_gradient_training.privacy import privacy_certified_training
from abstract_gradient_training.unlearning import unlearning_certified_training
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training import training_utils
from abstract_gradient_training import privacy_utils
from abstract_gradient_training import test_metrics
from abstract_gradient_training import bounded_models
from abstract_gradient_training import bounded_optimizers
from abstract_gradient_training import bounded_losses
from abstract_gradient_training import interval_arithmetic

logger = logging.getLogger("abstract_gradient_training")
logger.handlers.clear()
formatter = logging.Formatter("[AGT] [%(levelname)-8s] [%(asctime)s] %(message)s", datefmt="%H:%M:%S")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
