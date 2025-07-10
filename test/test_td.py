import numpy as np
import torch
from tensordict import TensorDict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import DataLoader

import pandas as pd

import matplotlib.pyplot as plt

def discretize_action(action, low=-1.0, high=1.0, bins=256):
    action_clipped = action.clamp(low, high - 1e-6)  # prevent upper-edge spill
    bin_width = (high - low) / bins
    return torch.floor((action_clipped - low) / bin_width).long()

def undiscretize_action(indices, low=-1.0, high=1.0, bins=256, gripper_index=6):
    bin_width = (high - low) / bins
    cont = low + (indices.float() + 0.5) * bin_width
    cont[..., gripper_index] = torch.where(
        indices[..., gripper_index] < bins // 2,
        torch.tensor(-1.0, device=indices.device),
        torch.tensor(1.0, device=indices.device)
    )
    return cont

def unnormalize(value, q01, q99):
    return (value+1) * 0.5 * (q99 - q01) + q01

def normalize(value, q01, q99):
    return torch.clamp(2 * (value - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)


def action_histogram(actions, num_bins=256):
    # actions: [N, D] where D is action dim
    hist = []
    for d in range(actions.shape[1]):
        h = np.bincount(actions[:, d], minlength=num_bins)
        hist.append(h)
    return np.stack(hist, axis=0)  # s

def plot_all_action_histograms(hist, save_path="action_histograms.png"):
    print(hist.shape)
    D = hist.shape[0]
    fig, axs = plt.subplots(D, 1, figsize=(10, 2 * D), sharex=True)
    for d in range(D):
        axs[d].bar(range(hist.shape[1]), hist[d])
        axs[d].set_title(f"Dimension {d}")
        axs[d].set_ylabel("Count")
    axs[-1].set_xlabel("Action Bin")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_bin_edges_np(actions, num_bins=256):
    # actions: [N, D]
    D = actions.shape[1]
    bin_edges = []
    for d in range(D):
        edges = np.percentile(actions[:, d], q=np.linspace(0, 100, num_bins + 1))
        bin_edges.append(edges)
    return np.stack(bin_edges)  # shape [D, num_bins + 1]

def discretize_np(actions, bin_edges):
    D = actions.shape[1]
    indices = np.zeros_like(actions, dtype=np.int64)
    for d in range(D):
        # np.digitize returns bin index: 1 to num_bins+1
        idx = np.digitize(actions[:, d], bin_edges[d], right=False) - 1
        # clamp to [0, num_bins - 1]
        indices[:, d] = np.clip(idx, 0, len(bin_edges[d]) - 2)
    return indices  # shape [N, D]

def undiscretize_np(indices, bin_edges):
    N, D = indices.shape
    values = np.zeros_like(indices, dtype=np.float32)
    for d in range(D):
        left = bin_edges[d][indices[:, d]]
        right = bin_edges[d][indices[:, d] + 1]
        values[:, d] = 0.5 * (left + right)
    return values  # shape [N, D]

import numpy as np

def histogram_equal_freq_discretize(actions, num_bins=256):
    """
    Discretize continuous actions using histogram binning with equal frequency (quantile bins).
    Args:
        actions: np.ndarray of shape [N, D], continuous values
        num_bins: number of bins (e.g., 256)

    Returns:
        discrete_actions: np.ndarray of shape [N, D] with integer bin indices
        bin_edges: np.ndarray of shape [D, num_bins + 1], the bin edges per dim
    """
    N, D = actions.shape
    bin_edges = np.zeros((D, num_bins + 1), dtype=np.float32)
    discrete_actions = np.zeros((N, D), dtype=np.int64)

    for d in range(D):
        # Compute quantile-based bin edges
        edges = np.percentile(actions[:, d], q=np.linspace(0, 100, num_bins + 1))
        # Fix duplicate edges to avoid collapsing bins
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        bin_edges[d] = edges

        # Digitize to get bin indices
        indices = np.digitize(actions[:, d], edges, right=False) - 1
        indices = np.clip(indices, 0, num_bins - 1)
        discrete_actions[:, d] = indices

    return discrete_actions, bin_edges

import matplotlib.pyplot as plt
import numpy as np

def plot_bin_counts_and_widths(discrete_actions, bin_edges, save_prefix=None):
    """
    Combine bin count and bin width plots into one figure per action dimension.
    """
    D = discrete_actions.shape[1]
    num_bins = bin_edges.shape[1] - 1

    for d in range(D):
        counts = np.bincount(discrete_actions[:, d], minlength=num_bins)
        widths = bin_edges[d, 1:] - bin_edges[d, :-1]

        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Top: bin counts
        axs[0].bar(range(num_bins), counts, color='tab:blue')
        axs[0].set_title(f"Bin Counts - Dim {d}")
        axs[0].set_xlabel("Bin Index")
        axs[0].set_ylabel("Count")

        # Bottom: bin widths
        axs[1].plot(widths, color='tab:orange')
        axs[1].set_title(f"Bin Widths - Dim {d}")
        axs[1].set_xlabel("Bin Index")
        axs[1].set_ylabel("Bin Width")

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_combined_dim{d}.png")
            plt.close()
        else:
            plt.show()


device = "cuda"
repo_id = "lerobot/libero_10_image"
data_path = "/network/projects/real-g-grp/libero_td/libero10_dinov2_base_patch_cont.pt"

train_dataset = TensorDict.load_memmap(data_path)

action_data = train_dataset["action"][:,0].float().numpy()


fig, axs = plt.subplots(1, 7, figsize=(20, 3))  # 1 row, 7 columns of plots
for i in range(7):
    axs[i].hist(action_data[:, i], bins=10, edgecolor='black', density=True)
    axs[i].set_title(f'Col {i}')
plt.tight_layout()
plt.savefig("normal_hist.png")
exit(0)
# ranks = pd.Series(actions).rank(method="first")  # Get unique ranks
# discrete = pd.qcut(ranks, q=256, labels=False, retbins=False).to_numpy()


# First, we download the tokenizer from the Hugging Face model hub
# Here, we will not use the pre-trained tokenizer weights, but only the source code
# to train a new tokenizer on our own data.

from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Load your action data for tokenizer training
# Chunks do not need to be of the same length, we will use dummy data
# action_data = np.random.rand(4000, 50, 14)

# Train the new tokenizer, depending on your dataset size this can take a few minutes
tokenizer = tokenizer.fit(action_data)

# Save the new tokenizer, optionally push it to the Hugging Face model hub
tokenizer.save_pretrained("libero_10_image_fast")
# tokenizer.push_to_hub("YourUsername/my_new_tokenizer")

label_list = []
for i in range(actions.shape[-1]):
    actions_i = actions[:,i]
    ranks = pd.Series(actions_i).rank(method="first")  # Get unique ranks
    print(ranks)
    discrete = pd.qcut(ranks, q=256, labels=False, retbins=False).to_numpy()
    label_list.append(discrete)

discrete_actions = np.stack(label_list, axis=-1)
# discrete_actions, bin_edges = histogram_equal_freq_discretize(actions, 256)
# bin_edges = compute_bin_edges_np(actions, 256)
# discrete = discretize_np(actions, bin_edges)
# print(discrete_actions.shape)
# plot_all_action_histograms(action_histogram(discrete_actions), "action_hist_4.png")
plot_all_action_histograms(action_histogram(discrete_actions), "rank_qcut_histogram")
# plot_bin_counts_and_widths(discrete_actions, bin_edges, save_prefix="action_hist5")
exit(0)
print(train_dataset["image"].contiguous().shape)

actions = train_dataset["action"].float()
states = train_dataset["state"].float()


action_q01 = torch.quantile(actions[:,0,:], q=0.01, dim=0)
action_q99 = torch.quantile(actions[:,0,:], q=0.99, dim=0)

state_q01 = torch.quantile(states, q=0.01, dim=0)
state_q99 = torch.quantile(states, q=0.99, dim=0)

# Create a TensorDict with some float tensors
metadata = TensorDict({
    "action": TensorDict({
        "Q01": action_q01,
        "Q99": action_q99,
    }, batch_size=[]),
    "state": TensorDict({
        "Q01": state_q01,
        "Q99": state_q99,
    }),
}, batch_size=[])

norm_actions = normalize(actions, action_q01.unsqueeze(0).unsqueeze(0), action_q99.unsqueeze(0).unsqueeze(0))
norm_states = normalize(states, state_q01.unsqueeze(0), state_q99.unsqueeze(0))

discrete_actions = discretize_action(norm_actions)

print(actions[:10, 0])
# print(discrete_actions[:50, 0, :-1])
print(unnormalize(undiscretize_action(discrete_actions), action_q01.unsqueeze(0).unsqueeze(0), action_q99.unsqueeze(0).unsqueeze(0))[:10, 0])

# print(norm_actions.mean(), norm_states.mean())
train_dataset["action"] = discrete_actions
train_dataset["state"] = norm_states.to(torch.bfloat16)

# print(train_dataset["action"].dtype)
# print(train_dataset["action"].mean(), train_dataset["state"].mean())
train_dataset.save(f"/network/projects/real-g-grp/libero_td/libero10_dinov2_base_reg_norm_discrete.pt", copy_existing=True)
# metadata.save(f"/network/projects/real-g-grp/libero10_metadata.pt")
