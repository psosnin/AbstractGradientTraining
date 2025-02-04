# %%
import copy

import torch
import torch.utils.data
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import script_utils
import octmnist_train


# %%
class poisoned_dataloader:
    """Dataloader that performs random flips on the labels of the Drusen class."""

    def __init__(self, dataloader, label_k_poison):
        self.dataloader = dataloader
        self.label_k_poison = label_k_poison
        self.poisoned_dataloader = self._poison_dataloader()

    def _poison_dataloader(self):
        poisoned_dataloader = []
        for x, y in self.dataloader:
            # randomly choose a subset of the data to poison
            idx = torch.randperm(len(y))[: self.label_k_poison]
            y[idx] = 1 - y[idx]
            poisoned_dataloader.append((x, y))
        return poisoned_dataloader

    def __iter__(self):
        return iter(self.poisoned_dataloader)

    def __len__(self):
        return len(self.poisoned_dataloader)


# %%
model = octmnist_train.get_pretrained_model()
dataset_drusen, test_dataset_drusen = octmnist_train.get_dataset(exclude_classes=[0, 1, 3])
dataset_clean, test_dataset_clean = octmnist_train.get_dataset(exclude_classes=[2])


# %%
cert_accs = []
poison_accs = []
k_poison_vals = list(range(0, 1001, 50))

for k in k_poison_vals:
    conf = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
    conf.label_k_poison = k
    # clean run
    torch.manual_seed(octmnist_train.SEED)
    dl_train = torch.utils.data.DataLoader(dataset_drusen, batch_size=octmnist_train.DRUSEN_BATCHSIZE, shuffle=True)
    dl_train_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=octmnist_train.CLEAN_BATCHSIZE, shuffle=True)
    conv_layers = agt.bounded_models.IntervalBoundedModel(model[0:5], trainable=False)
    bounded_model = agt.bounded_models.IntervalBoundedModel(model[5:], transform=conv_layers)
    agt.poison_certified_training(bounded_model, conf, dl_train, dl_clean=dl_train_clean)
    cert_accs.append(agt.test_metrics.test_accuracy(bounded_model, *test_dataset_drusen.tensors, epsilon=0)[0])

    # poisoned run
    torch.manual_seed(octmnist_train.SEED)
    dl_train = torch.utils.data.DataLoader(dataset_drusen, batch_size=octmnist_train.DRUSEN_BATCHSIZE, shuffle=True)
    dl_train = poisoned_dataloader(dl_train, k)
    dl_train_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=octmnist_train.CLEAN_BATCHSIZE, shuffle=True)
    conv_layers = agt.bounded_models.IntervalBoundedModel(model[0:5], trainable=False)
    bounded_model = agt.bounded_models.IntervalBoundedModel(model[5:], transform=conv_layers)
    agt.poison_certified_training(bounded_model, conf, dl_train, dl_clean=dl_train_clean)
    poison_accs.append(agt.test_metrics.test_accuracy(bounded_model, *test_dataset_drusen.tensors, epsilon=0)[1])

# %%
"""Plot the results."""

subplots = (1, 1)
fig, ax = plt.subplots(*subplots, layout="constrained", dpi=300)

colors = iter(script_utils.colours.values())
nom_accs = [cert_accs[0] for _ in cert_accs]
color = next(colors)
ax.plot(k_poison_vals, cert_accs, color=color, label="Certified Accuracy")
ax.plot(k_poison_vals, nom_accs, color=color, linestyle="--", label="Clean Accuracy")
color = next(colors)
ax.plot(k_poison_vals, poison_accs, color=color, label="Poisoned Accuracy", linestyle=":")

ax.set_ylabel("Accuracy")
ax.set_xlabel("Attack Size ($m$)")
ax.legend(loc="lower left", fontsize="x-small", handlelength=1.3)

_, _, _, fig_dir = script_utils.make_dirs()
script_utils.apply_figure_size(fig, script_utils.set_size(0.3, subplots, shrink_height=1.5), dpi=300)
plt.savefig(f"{fig_dir}/oct_mnist_rand_flips.pdf", dpi=300)
