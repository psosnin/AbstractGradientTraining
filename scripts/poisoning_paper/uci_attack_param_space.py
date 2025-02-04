# %%
"""Plot a heuristic poisoning attack in parameter space."""

import copy
import itertools
from collections import defaultdict

import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches
import matplotlib.colors

import abstract_gradient_training as agt

import train_uci
import script_utils

# %%
""" Configure the training and attack parameteters. """
torch.manual_seed(train_uci.SEED)
batchsize = 10000
dl_train, _ = train_uci.get_dataset(batchsize)
dl_train = list(itertools.islice(dl_train, 30))  # only going to plot the first 30 iterations

model = train_uci.get_model()
config = agt.AGTConfig(
    fragsize=10000,
    lr_decay=0.5,
    learning_rate=0.02,
    lr_min=0.001,
    k_poison=1000,
    label_k_poison=1000,
    epsilon=0.02,
    label_epsilon=0.05,
    n_epochs=1,
    device="cuda:0",
    loss="mse",
    paired_poison=True,
)
test_point = next(iter(dl_train))[0][0].to(config.device)

# %%
"""Helper functions for the poisoning attack. """


def craft_poison(bounded_model, bounded_loss, examples, labels, objective, epsilon, label_epsilon, max_iter=20):
    """Craft a poison within the epsilon ball of examples that maximises the poisoning objective using PGD."""
    x0 = examples.clone().detach()
    y = labels.clone().detach()
    y.requires_grad = True
    x0.requires_grad = True
    for _ in range(max_iter):
        optimizer = torch.optim.Adam([x0, y], lr=0.1)  # type: ignore
        logits_n = bounded_model.forward(x0, retain_intermediate=True)
        dl = bounded_loss.backward(logits_n, y)
        grads = bounded_model.backward(dl)
        obj = objective(grads)
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()
        # project back to linf ball
        x0.data = x0.clamp(examples - epsilon, examples + epsilon)
        y.data = y.data.clamp(labels - label_epsilon, labels + label_epsilon)
    return x0, y


def train_poison(model, config, poison_objective):
    """Train the model using a poisoning attack with the given objective."""
    poison_model = copy.deepcopy(model)
    bounded_model = agt.bounded_models.IntervalBoundedModel(poison_model).to(config.device)
    bounded_loss = agt.bounded_losses.BoundedMSELoss(reduction="none")
    poison_param_list = []
    iter_count = 0
    for _ in range(config.n_epochs):
        for x, y in dl_train:
            poison_param_list.append([p.clone().detach() for p in bounded_model.param_n])
            # learning rate decay as in AGT
            lr = max(config.learning_rate / (1 + config.lr_decay * iter_count), config.lr_min)
            iter_count += 1
            if x.size(0) != batchsize:
                continue
            x, y = x.to(config.device), y.to(config.device)
            # poison config.k_poison examples
            selected = torch.randperm(x.size(0))[: config.k_poison]
            poison_x, poison_y = craft_poison(
                bounded_model,
                bounded_loss,
                x[selected],
                y[selected],
                poison_objective,
                config.epsilon,
                config.label_epsilon,
            )
            # replace the examples with the poisoned examples
            with torch.no_grad():
                x[selected] = poison_x
                y[selected] = poison_y
                logits_n = bounded_model.forward(x, retain_intermediate=True)
                dl = bounded_loss.backward(logits_n, y)
                grads_n = bounded_model.backward(dl)
                grads_n = [g.clamp(-config.clip_gamma, config.clip_gamma) for g in grads_n]
                grads_n = [g.mean(dim=0) for g in grads_n]
                for p, g in zip(bounded_model.param_n, grads_n):
                    p -= lr * g
    return poison_param_list


def get_poison_trajectories(model, config, layer, idx_1, idx_2):
    """
    Get the poisoned training trajectories for a range of different poisoning objectives.
    Each of the poisoning objectives aims to nudge the training as far in a given direction of parameter space
    during training.
    """
    objectives = [
        lambda grads: grads[layer].sum(dim=0).flatten()[idx_1],
        lambda grads: grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: grads[layer].sum(dim=0).flatten()[idx_1] + grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: grads[layer].sum(dim=0).flatten()[idx_1] - grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: -grads[layer].sum(dim=0).flatten()[idx_1] - grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: -grads[layer].sum(dim=0).flatten()[idx_1] + grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: -grads[layer].sum(dim=0).flatten()[idx_1] * grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: grads[layer].sum(dim=0).flatten()[idx_1] * grads[layer].sum(dim=0).flatten()[idx_2],
        lambda grads: -grads[layer].sum(dim=0).flatten()[idx_1],
        lambda grads: -grads[layer].sum(dim=0).flatten()[idx_2],
    ]

    poison_trajectories = []
    for objective in tqdm.tqdm(objectives):
        poison_param_list = train_poison(model, config, objective)
        poison_trajectories.append(poison_param_list)
    return poison_trajectories


def get_agt_trajectory(model, config):
    """Get the training bounds for AGT under the poisoning attack."""
    bounded_model = agt.bounded_models.IntervalBoundedModel(model).to(config.device)
    results = []

    def log(bm):
        """Log the loss bounds at this iteration."""
        pl = [p.detach().flatten().clone().cpu().numpy() for p in bm.param_l]
        pn = [p.detach().flatten().clone().cpu().numpy() for p in bm.param_n]
        pu = [p.detach().flatten().clone().cpu().numpy() for p in bm.param_u]
        results.append([pl, pn, pu])

    config.on_iter_start_callback = log

    agt.poison_certified_training(bounded_model, config, dl_train)
    return results


# %%

""" Run the poisoning attacks."""

target_indices = [  # (layer, idx_1, idx_2)
    (2, 7, 4),
    (2, 15, 16),
    (2, 33, 17),
    (2, 0, 4),
]


poison_trajectories = defaultdict(list)

for layer, i, j in target_indices:
    if (layer, i, j) not in poison_trajectories:
        if (layer, j, i) in poison_trajectories:
            poison_trajectories[(layer, i, j)] = poison_trajectories[(layer, j, i)]
        else:
            poison_trajectories[(layer, i, j)].extend(get_poison_trajectories(model, config, layer, i, j))

# %%
"""Run the certified training."""

agt_training = get_agt_trajectory(model, config)

# %%
"""Plot the certified bounds along with the poisoned trajectories in parameter space."""


def plot_params(training, layer, idx_1, idx_2, ax, step=5, poisoned_trajectories=None):
    """
    Plot the parameter space training trajectory for parameters at idx_1 and idx_2 in the given layer.
    """
    p_l = [t[0][layer] for t in training]
    p_n = [t[1][layer] for t in training]
    p_u = [t[2][layer] for t in training]
    L = len(p_l)
    palette = sns.color_palette("rocket", n_colors=L // step + 1)

    for k in range(0, L, step):
        x_l = p_l[k][idx_1]
        x_u = p_u[k][idx_1]
        y_l = p_l[k][idx_2]
        y_u = p_u[k][idx_2]
        box = matplotlib.patches.Rectangle(
            (x_l, y_l), x_u - x_l, y_u - y_l, edgecolor=palette[k // step], alpha=0.5, facecolor="none"
        )
        ax.add_patch(box)

    # plot the poisoned trajectories
    if poisoned_trajectories:
        for t in poisoned_trajectories:
            ax.plot(
                [p[layer].flatten()[idx_1].cpu().detach().numpy() for p in t],
                [p[layer].flatten()[idx_2].cpu().detach().numpy() for p in t],
                color="black",
                linestyle="--",
            )
    ax.plot(
        [p_n[k][i] for k in range(L)],
        [p_n[k][j] for k in range(L)],
        color="red",
    )
    ax.set_box_aspect(1)


subplots = (1, max(4, len(target_indices)))
_, _, _, fig_dir = script_utils.make_dirs()

fig, ax = plt.subplots(*subplots, layout="constrained", dpi=300)

for (layer, i, j), a in zip(target_indices, ax):
    a.axes.set_aspect("equal")
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    plot_params(agt_training, layer, i, j, a, 1, poison_trajectories[(layer, i, j)])
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(agt_training))
sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.28, pad=0.01, label="Training Iteration")

ax[0].plot([], [], color="red", label="Clean")
ax[0].plot([], [], color="black", label="Poisoned", linestyle="--")
ax[0].legend()

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=2.2), dpi=300)
plt.savefig(f"{fig_dir}/uci_param_space_poison.pdf", dpi=300)
