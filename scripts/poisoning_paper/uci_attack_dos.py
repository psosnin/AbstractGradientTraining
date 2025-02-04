# %%
import copy
import itertools

import torch
import matplotlib.pyplot as plt

import abstract_gradient_training as agt

import train_uci
import script_utils

# %%

seed = 0
batchsize = 10000
target_idx = 560  # 2
dl_train, test_loader = train_uci.get_dataset(batchsize)
dl_train = list(itertools.islice(dl_train, 150))
test_batch, test_label = next(iter(test_loader))
dl_test = itertools.cycle([(test_batch[target_idx], test_label[target_idx])])  # select this as our poisoning target
torch.manual_seed(seed)
# warning: there were some bugs occuring when using a different GPU than the one specified in train_uci.py
# change the device from "cuda:0" to "cuda:1" at your own risk
config = agt.AGTConfig(
    fragsize=10000,
    learning_rate=0.1,
    lr_decay=2.0,
    lr_min=0.001,
    n_epochs=1,
    device="cuda:0",
    loss="mse",
    clip_gamma=0.06,
)

model = train_uci.get_model(1, 64, seed)

n_poison_trajectories = 6

# %%

"""Simple heuristic feature collision poisoning attack."""


@torch.no_grad()
def train_poison(model, config, ptype, r):
    """Train the model under a heuristic poisoning attack."""

    model = copy.deepcopy(model)
    config = copy.deepcopy(config)
    bounded_model = agt.bounded_models.IntervalBoundedModel(model).to(config.device)
    criterion = config.get_bounded_loss_fn()
    n_poison = config.k_private

    target_point, target_label = next(dl_test)
    target_point, target_label = target_point.to(config.device), target_label.to(config.device)

    poison_training = []
    iter_count = 0
    for _ in range(config.n_epochs):
        for x, y in dl_train:
            # learning rate decay as in AGT
            lr = max(config.learning_rate / (1 + config.lr_decay * iter_count), config.lr_min)
            iter_count += 1
            # check the model performance on the target point and store the result.
            y_test = bounded_model.forward(target_point)
            poison_training.append(criterion.forward(y_test, target_label).mean().item())
            if x.size(0) != batchsize:
                continue
            # print(f"Test MSE: {standard_training[-1]:.4g}")
            x, y = x.to(config.device).clone(), y.to(config.device).clone()
            # poison config.k_poison examples
            selected = torch.randperm(x.size(0))[:n_poison]
            # replace the selected points with the features of the target point
            poison_x = torch.tile(target_point, (n_poison, 1))
            if ptype == 1:  # replace the selected labels with random noise
                poison_y = torch.randn(n_poison, 1, device=config.device) * 100 * r
            elif ptype == 2:  # replace the selected labels with the target label
                poison_y = torch.tile(target_label, (n_poison, 1))
            elif ptype == 3:  # replace the selected labels with a constant
                poison_y = torch.ones(n_poison, 1, device=config.device) * r
            else:
                raise ValueError("Invalid poisoning type")
            # perform the poisoning
            x[selected], y[selected] = poison_x, poison_y
            # perform the training step (with clipping)
            logits_n = bounded_model.forward(x, retain_intermediate=True)
            dl = criterion.backward(logits_n, y)
            grads_n = bounded_model.backward(dl)
            grads_n = [g.clamp(-config.clip_gamma, config.clip_gamma) for g in grads_n]
            grads_n = [g.mean(dim=0) for g in grads_n]
            for p, g in zip(bounded_model.param_n, grads_n):
                p -= lr * g
    return poison_training


def get_poison_trajectories(model, config):
    """Run a series of different poisoning attacks."""
    trajectories = []
    for i in range(n_poison_trajectories):
        poison_training = train_poison(model, config, ptype=(i % 3) + 1, r=i)
        trajectories.append(poison_training)
    return trajectories


def get_agt_trajectory(model, config):
    """Get the training bounds for AGT under the poisoning attack."""
    bounded_model = agt.bounded_models.IntervalBoundedModel(model).to(config.device)
    results = []
    target_point, target_label = next(dl_test)
    target_point, target_label = target_point.to(config.device), target_label.to(config.device)

    def log(bm):
        """Log the loss bounds at this iteration."""
        results.append(agt.test_metrics.test_mse(bm, target_point, target_label))

    config.on_iter_start_callback = log

    agt.privacy_certified_training(bounded_model, config, dl_train, dl_test)
    return results


# %%

"""Run the poisoning attacks and agt bounds."""

k_private_1 = 100
config.k_private = k_private_1
agt_training_1 = get_agt_trajectory(model, config)
trajectories_1 = get_poison_trajectories(model, config)

k_private_2 = 200
config.k_private = k_private_2
trajectories_2 = get_poison_trajectories(model, config)
agt_training_2 = get_agt_trajectory(model, config)

k_private_3 = 400
config.k_private = k_private_3
trajectories_3 = get_poison_trajectories(model, config)
agt_training_3 = get_agt_trajectory(model, config)


# %%

"""Plot the results."""


def plot_trajectories(ax, agt_trajectory, poisoned_trajectories, k_private):

    mse_u = [t[0] for t in agt_trajectory]
    mse_n = [t[1] for t in agt_trajectory]
    mse_l = [t[2] for t in agt_trajectory]

    ax.fill_between(range(len(mse_n)), mse_l, mse_u, color="red", alpha=0.3, lw=0)

    ax.plot(mse_n, "red")
    ax.set_title(f"$n={k_private}$", fontsize="medium")

    for t in poisoned_trajectories:
        ax.plot(t, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 150)


subplots = (1, 3)
_, _, _, fig_dir = script_utils.make_dirs()

fig, ax = plt.subplots(
    *subplots,
    figsize=script_utils.set_size(1.0, subplots, shrink_height=1.2),
    layout="constrained",
    sharey=True,
    dpi=300,
)
fig.supxlabel("Training Iteration", fontsize="medium")
fig.supylabel(r"$\left[y^{\text{target}} - f^\theta(x^{\text{target}})\right]^2$", fontsize="medium")
ax[0].plot([], [], color="red", label="Clean")
ax[0].plot([], [], color="black", label="Poisoned", linestyle="--")
ax[0].legend(loc="upper right")

plot_trajectories(ax[0], agt_training_1, trajectories_1, k_private_1)
plot_trajectories(ax[1], agt_training_2, trajectories_2, k_private_2)
plot_trajectories(ax[2], agt_training_3, trajectories_3, k_private_3)

script_utils.apply_figure_size(fig, script_utils.set_size(1.0, subplots, shrink_height=1.3), dpi=300)
plt.savefig(f"{fig_dir}/uci_dos_attack.pdf", dpi=300)
