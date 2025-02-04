"""
Train a model on the OCT-MNIST dataset using abstract gradient training for a wide range of `k_poison` values.
We use the following setting:
- The model is first pre-trained on the dataset with the drusen class removed, i.e. trained to distinguish normal vs.
    (choroidal neovascularization, diabetic macular edem) classes.
- We'll then fine-tune the model on the full dataset (including the drusen class) using Abstract Gradient Training for
    poison-safe certification.
- The model is a convolutional network with 3 convolutional layers and 2 fully connected layers. The pre-training will
    train all the layers, while the fine-tuning only trains the dense layers using AGT.
"""

# %%
import os

import torch
import torch.utils.data
import torchvision
import tqdm

from medmnist import OCTMNIST  # python -m pip install git+https://github.com/MedMNIST/MedMNIST.git

import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig

import robust_regularization
import script_utils

USE_CACHED = True  # whether to use previously cached results or to run from scratch
SEED = 1
# pretraining parameters
PT_BATCHSIZE = 100
PT_N_EPOCHS = 10
PT_LEARNING_RATE = 0.001
PT_EPSILON = 0.5
PT_MODEL_EPSILON = 0.001
PT_REG_STRENGTH = 0.3

NOMINAL_CONFIG = AGTConfig(
    fragsize=2000,
    learning_rate=0.05,
    n_epochs=2,
    device="cuda:0",
    loss="binary_cross_entropy",
    lr_decay=5.0,
    lr_min=0.001,
    log_level="WARNING",
)

CLEAN_BATCHSIZE = 3000
DRUSEN_BATCHSIZE = 3000


def get_dataset(exclude_classes=None, balanced=False):
    """
    Get OCT MedMNIST dataset as a binary classification problem of class 3 (normal) vs classes 0, 1, 2.
    """

    # get the datasets
    train_dataset = OCTMNIST(split="train", transform=torchvision.transforms.ToTensor())
    test_dataset = OCTMNIST(split="test", transform=torchvision.transforms.ToTensor())
    train_imgs, train_labels = train_dataset.imgs, train_dataset.labels
    test_imgs, test_labels = test_dataset.imgs, test_dataset.labels

    # filter out excluded classes
    if exclude_classes is not None:
        for e in exclude_classes:
            train_imgs = train_imgs[(train_labels != e).squeeze()]
            train_labels = train_labels[(train_labels != e).squeeze()]
            test_imgs = test_imgs[(test_labels != e).squeeze()]
            test_labels = test_labels[(test_labels != e).squeeze()]

    # convert to a binary classification problem
    train_labels = train_labels != 3  # i.e. 0 = normal, 1 = abnormal
    test_labels = test_labels != 3

    # apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1) / 255
    test_imgs = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1) / 255
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    # balance the training dataset such that the number of samples in each class is equal
    if balanced:
        n_ones = int(train_labels.sum().item())
        n_zeros = len(train_labels) - n_ones
        n_samples = min(n_ones, n_zeros)
        # find the indices of the ones, and then randomly sample n_samples from them
        idx_ones = torch.where(train_labels == 1)[0]
        ones_selection = torch.randperm(n_ones)
        idx_ones = idx_ones[ones_selection][:n_samples]
        # find the indices of the zeros, and then randomly sample n_samples from them
        idx_zeros = torch.where(train_labels == 0)[0]
        zeros_selection = torch.randperm(n_zeros)
        idx_zeros = idx_zeros[zeros_selection][:n_samples]
        idx = torch.cat([idx_ones, idx_zeros])
        train_imgs, train_labels = train_imgs[idx], train_labels[idx]

    # form dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_imgs, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_imgs, test_labels)
    return train_dataset, test_dataset


def get_pretrained_model(epsilon=PT_EPSILON, model_epsilon=PT_MODEL_EPSILON, reg_strength=PT_REG_STRENGTH):
    """
    Get the pretrained OCT MedMNIST model, with optional robust (advesarial) regularisation.
    If it doesn't exist, pretrain one.
    """
    _, model_dir, _, _ = script_utils.make_dirs()
    model_path = f"{model_dir}/medmnist_{SEED=}_{epsilon=}_{model_epsilon=}_{reg_strength=}.ckpt"
    torch.manual_seed(SEED)
    device = torch.device(NOMINAL_CONFIG.device)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 4, 2, 0),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 4, 1, 0),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(3200, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
    ).to(device)
    # check if a pre-trained model exists
    if os.path.exists(model_path) and USE_CACHED:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model = model.to(device)
        return model

    # else train the model
    dataset_pretrain, _ = get_dataset(exclude_classes=[2], balanced=True)
    dl_pretrain = torch.utils.data.DataLoader(dataset_pretrain, batch_size=PT_BATCHSIZE, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PT_LEARNING_RATE)  # type: ignore
    progress_bar = tqdm.trange(PT_N_EPOCHS, desc="Epoch")
    for _ in progress_bar:
        for i, (x, u) in enumerate(dl_pretrain):
            # Forward pass
            u, x = u.to(device), x.to(device)
            output = model(x)
            bce_loss = criterion(output.squeeze().float(), u.squeeze().float())
            if reg_strength > 0:
                regularization = robust_regularization.parameter_gradient_interval_regularizer(
                    model, x, u, "binary_cross_entropy", epsilon, model_epsilon
                )
            else:
                regularization = torch.tensor(0.0)
            loss = bce_loss + reg_strength * regularization  # type: ignore
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                progress_bar.set_postfix(bce_loss=bce_loss.item(), reg=regularization.item())  # type: ignore
    # save the model
    with open(model_path, "wb") as file:
        torch.save(model.state_dict(), file)

    return model


def run_with_config(config, model, dataset_drusen, dataset_clean, privacy=True):
    """If results for this configuration are already computed, load them from disk. Otherwise, run the certified
    training using AGT, then save and return the results."""
    results_dir, _, _, _ = script_utils.make_dirs()
    fname = f"{results_dir}/medmnist_{config.hash()}"
    conv_bounded_model = agt.bounded_models.IntervalBoundedModel(model[0:5], trainable=False)
    bounded_model = agt.bounded_models.IntervalBoundedModel(model[5:], transform=conv_bounded_model)
    if os.path.isfile(fname) and USE_CACHED:  # run exists, so return the previous results
        bounded_model.load_params(fname)
    else:
        # check whether the given config should be either unlearning or privacy training
        assert not (config.k_unlearn and config.k_private)
        torch.manual_seed(SEED)
        dl_train = torch.utils.data.DataLoader(dataset_drusen, batch_size=DRUSEN_BATCHSIZE, shuffle=True)
        dl_train_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=CLEAN_BATCHSIZE, shuffle=True)
        if privacy:
            assert config.k_poison == 0
            agt.privacy_certified_training(bounded_model, config, dl_train, dl_public=dl_train_clean)
        else:
            assert config.k_private == 0
            agt.poison_certified_training(bounded_model, config, dl_train, dl_clean=dl_train_clean)
        bounded_model.save_params(fname)
    return bounded_model


# %%
if __name__ == "__main__":
    torch.manual_seed(1)
    epsilon = 0.01
    _, dataset_test = get_dataset(exclude_classes=[2], balanced=True)
    test_batch, test_labels = dataset_test.tensors
    standard_model = get_pretrained_model(0, 0, 0)
    # split the model into convolutional and linear layers and wrap in IntervalBoundedModel
    conv_layers = agt.bounded_models.IntervalBoundedModel(standard_model[0:5])
    bounded_model = agt.bounded_models.IntervalBoundedModel(standard_model[5:], transform=conv_layers)
    accs = agt.test_metrics.test_accuracy(
        bounded_model,
        test_batch,
        test_labels,
        epsilon=epsilon,
    )
    accs = ", ".join([f"{a:.2f}" for a in accs])

    print(f"Accuracy of non-robustly trained classifier on test set with epsilon={epsilon}: [{accs}]")

    robust_model = get_pretrained_model()
    conv_layers = agt.bounded_models.IntervalBoundedModel(robust_model[0:5])
    bounded_model = agt.bounded_models.IntervalBoundedModel(robust_model[5:], transform=conv_layers)
    accs = agt.test_metrics.test_accuracy(
        bounded_model,
        test_batch,
        test_labels,
        epsilon=epsilon,
    )
    accs = ", ".join([f"{a:.2f}" for a in accs])

    print(f"Accuracy of robustly trained classifier on test set with epsilon={epsilon}: [{accs}]")
