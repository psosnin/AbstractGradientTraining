# %%
import sys
import copy

import tqdm
import torch
import torch.utils.data
import matplotlib.pyplot as plt

import abstract_gradient_training as agt
import labellines

import octmnist_train
import script_utils

# %%
torch.manual_seed(0)

# get the test datasets
dataset_drusen, test_dataset_drusen = octmnist_train.get_dataset(exclude_classes=[0, 1, 3])
dataset_clean, test_dataset_clean = octmnist_train.get_dataset(exclude_classes=[2])
dataset_all, test_dataset_all = octmnist_train.get_dataset()

# get the pretrained model
pretrained_model = octmnist_train.get_pretrained_model()
conv_layers = agt.bounded_models.IntervalBoundedModel(pretrained_model[0:5], trainable=False)
bounded_model = agt.bounded_models.IntervalBoundedModel(pretrained_model[5:], transform=conv_layers)

# evaluate the pre-trained model
drusen_acc = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_drusen.tensors, epsilon=0)
pretrained_acc = drusen_acc[1]
clean_acc = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_clean.tensors, epsilon=0)
all_acc = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_all.tensors, epsilon=0)

print("=========== Pre-trained model accuracy ===========", file=sys.stderr)
print(f"Class 2 (Drusen) : nominal = {drusen_acc[1]:.2g}", file=sys.stderr)
print(f"Class 2 (Drusen) : backdoor = {drusen_acc[1]:.2g}", file=sys.stderr)
print(f"Classes 0, 1, 3  : nominal = {clean_acc[1]:.2g}", file=sys.stderr)
print(f"All Classes      : nominal = {all_acc[1]:.2g}", file=sys.stderr)

# fine-tune the model using abstract gradient training (keeping the convolutional layers fixed)
conf = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
conf.log_level = "DEBUG"
conf.k_poison = 50
conf.epsilon = 0.01
dl_train = torch.utils.data.DataLoader(dataset_drusen, batch_size=octmnist_train.DRUSEN_BATCHSIZE, shuffle=True)
dl_train_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=octmnist_train.CLEAN_BATCHSIZE, shuffle=True)
dl_test_drusen = torch.utils.data.DataLoader(test_dataset_drusen, batch_size=1000, shuffle=False)
agt.poison_certified_training(bounded_model, conf, dl_train, dl_test_drusen, dl_train_clean)

# evaluate the fine-tuned model
drusen_acc = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_drusen.tensors, epsilon=0)
clean_acc = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_clean.tensors, epsilon=0)
all_acc = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_all.tensors, epsilon=0)

print("=========== Fine-tuned model accuracy + bounds ===========", file=sys.stderr)
print(f"Class 2 (Drusen) : nominal = {drusen_acc[1]:.2g}, certified bound = {drusen_acc[0]:.2g}", file=sys.stderr)
print(f"Classes 0, 1, 3  : nominal = {clean_acc[1]:.2g}, certified bound = {clean_acc[0]:.2g}", file=sys.stderr)
print(f"All Classes      : nominal = {all_acc[1]:.2g}, certified bound = {all_acc[0]:.2g}", file=sys.stderr)

# %%
"""Run sweeps over various values of k_poison. """
k_poisons = list(range(0, 601, 20))

print("1. Feature poisoning with bounded adversary")
config = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
epsilons = [0.01, 0.02, 0.1]
results_1 = {}
for epsilon in epsilons:
    results_1[epsilon] = {}
    for k in tqdm.tqdm(k_poisons):
        config.k_poison = k
        config.epsilon = epsilon
        bounded_model = octmnist_train.run_with_config(
            config, pretrained_model, dataset_drusen, dataset_clean, privacy=False
        )
        results_1[epsilon][k] = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_drusen.tensors, epsilon=0)

print("2. Label flipping with bounded adversary")
config = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
results_2 = {}
for k in tqdm.tqdm(k_poisons):
    config.label_k_poison = k
    bounded_model = octmnist_train.run_with_config(
        config, pretrained_model, dataset_drusen, dataset_clean, privacy=False
    )
    results_2[k] = agt.test_metrics.test_accuracy(bounded_model, *test_dataset_drusen.tensors, epsilon=0)

print("3. Feature and label poisoning with unbounded adversary")
config = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
clip_gammas = [0.5, 1.0, 4.0]
results_3 = {}
for clip_gamma in clip_gammas:
    results_3[clip_gamma] = {}
    for k in tqdm.tqdm(k_poisons):
        config.k_private = k
        config.clip_gamma = clip_gamma
        bounded_model = octmnist_train.run_with_config(
            config, pretrained_model, dataset_drusen, dataset_clean, privacy=True
        )
        results_3[clip_gamma][k] = agt.test_metrics.test_accuracy(
            bounded_model, *test_dataset_drusen.tensors, epsilon=0
        )

print("4. Feature poisoning with bounded adversary, backdoor attack")
config = copy.deepcopy(octmnist_train.NOMINAL_CONFIG)
epsilons = [0.003, 0.006, 0.009]
results_4 = {}
for epsilon in epsilons:
    results_4[epsilon] = {}
    for k in tqdm.tqdm(k_poisons):
        config.k_poison = k
        config.epsilon = epsilon
        bounded_model = octmnist_train.run_with_config(
            config, pretrained_model, dataset_drusen, dataset_clean, privacy=False
        )
        results_4[epsilon][k] = agt.test_metrics.test_accuracy(
            bounded_model, *test_dataset_drusen.tensors, epsilon=epsilon
        )


# %%
""" Plot the results. """

# set plotting context
tex_fonts = {
    "axes.labelsize": "x-small",
    "xtick.labelsize": "x-small",
    "ytick.labelsize": "x-small",
}

plt.rcParams.update(tex_fonts)

subplots = (1, 4)

fig, axs = plt.subplots(*subplots, layout="constrained", dpi=300)

axs[0].set_title("Feature Poisoning\n(Bounded Adversary)", fontsize="x-small")
axs[0].set_xlabel("Attack Size ($n$)", fontsize="small")
colours = iter(script_utils.colours.values())
for i, (epsilon, results) in enumerate(results_1.items()):
    nom_accs = [res[1] for res in results.values()]
    cert_accs = [res[0] for res in results.values()]
    k_poisons = list(results.keys())
    color = next(colours)
    if i == 0:
        axs[0].plot(k_poisons, nom_accs, linestyle="--", color=color)
    axs[0].plot(k_poisons, cert_accs, label=rf"$\epsilon={epsilon}$", color=color)
labellines.labelLines(
    axs[0].get_lines(), align=False, drop_label=True, xvals=[460, 400, 280], fontsize="x-small", outline_width=2
)


axs[1].set_title("Label Poisoning\n(Bounded Adversary)", fontsize="x-small")
axs[1].set_xlabel("Attack Size ($n$)", fontsize="small")
color = next(iter(script_utils.colours.values()))
nom_accs = [res[1] for res in results_2.values()]
k_poisons = list(results_2.keys())
cert_accs = [res[0] for res in results_2.values()]
axs[1].plot(k_poisons, nom_accs, color=color, linestyle="--")
axs[1].plot(k_poisons, cert_accs, color=color)


axs[2].set_title("Feature + Label Poisoning\n(Unbounded Adversary)", fontsize="x-small")
axs[2].set_xlabel("Attack Size ($n$)", fontsize="small")
colors = iter(script_utils.colours.values())
for clip_gamma, results in results_3.items():
    nom_accs = [res[1] for res in results.values()]
    cert_accs = [res[0] for res in results.values()]
    k_poisons = list(results.keys())
    color = next(colors)
    axs[2].plot(k_poisons, nom_accs, color=color, linestyle="--")
    axs[2].plot(k_poisons, cert_accs, label=rf"$\kappa={clip_gamma}$", color=color)
labellines.labelLines(
    axs[2].get_lines(), align=False, drop_label=True, fontsize="x-small", xvals=[500, 260, 75], outline_width=2
)

axs[3].set_title("Feature Poisoning\n(Bounded Adversary)", fontsize="x-small")
axs[3].set_xlabel("Attack Size ($n$)", fontsize="small")
colors = iter(script_utils.colours.values())
for epsilon, results in results_4.items():
    cert_accs = [res[0] for res in results.values()]
    nom_accs = [cert_accs[0] for _ in results.values()]
    k_poisons = list(results.keys())
    color = next(colors)
    axs[3].plot(k_poisons, nom_accs, color=color, linestyle="--")
    axs[3].plot(k_poisons, cert_accs, label=rf"$\epsilon={epsilon}$", color=color)
labellines.labelLines(
    axs[3].get_lines(), align=False, drop_label=True, fontsize="x-small", xvals=[440, 370, 260], outline_width=2
)


axs[1].plot([], [], color="grey", label="Fine-tuned accuracy", linestyle="--")
for i in range(4):
    if i in [1, 2]:
        axs[i].set_yticklabels([])
    axs[i].set_xticks([0, 200, 400, k_poisons[-1]])
    axs[i].set_ylim(0, 1.0)
    axs[i].set_xlim(0, k_poisons[-1])
    if i < 3:
        axs[i].axhline(pretrained_acc, label="Pre-trained accuracy", color=script_utils.lb_color, linestyle="-.")
axs[0].set_ylabel("Certified Accuracy", fontsize="small")
axs[3].set_ylabel("Backdoor Accuracy", fontsize="small")
axs[1].legend(loc="lower center", fontsize="xx-small", handlelength=2.0, labelspacing=0.2, borderpad=0.25)

_, _, _, fig_dir = script_utils.make_dirs()
script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=1.5), dpi=300)
plt.savefig(f"{fig_dir}/octmnist_train_poisoning.pdf", dpi=300)
