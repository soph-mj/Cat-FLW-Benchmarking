"""
eval_utils.py

Metrics and visualization for comparing landmark detection approaches.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_nrmse(pred, target, image_size=224):
    """
    Normalized RMSE as a percentage of image size -- lets you compare directly
    against published benchmarks (e.g. the FGS paper's ~16.76% for ShuffleNetV2
    on a similar landmark task).
    pred, target: (N_LANDMARKS, 2) or (batch, N_LANDMARKS, 2)
    """
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return (rmse / image_size).item() * 100


def compute_nme(pred, target, inter_ocular_dist):
    """
    Normalized Mean Error, normalized by inter-ocular distance rather than
    image size -- the standard metric in most facial landmark literature
    (more robust to face scale/crop variation than NRMSE).
    pred, target: (batch, N_LANDMARKS, 2); inter_ocular_dist: (batch,)
    """
    per_point_error = torch.norm(pred - target, dim=-1)  # (batch, N_LANDMARKS)
    mean_error_per_sample = per_point_error.mean(dim=-1)  # (batch,)
    nme = (mean_error_per_sample / inter_ocular_dist).mean().item()
    return nme * 100  # as percentage


def visualize_predictions(image_tensor, pred_landmarks, gt_landmarks, save_path=None, title=None):
    """
    image_tensor: (3, H, W) torch tensor, values in [0, 1]
    pred_landmarks, gt_landmarks: (N_LANDMARKS, 2) numpy arrays or tensors
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    pred = pred_landmarks.detach().cpu().numpy() if torch.is_tensor(pred_landmarks) else pred_landmarks
    gt = gt_landmarks.detach().cpu().numpy() if torch.is_tensor(gt_landmarks) else gt_landmarks

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.scatter(gt[:, 0], gt[:, 1], c="lime", s=15, label="ground truth", marker="o")
    plt.scatter(pred[:, 0], pred[:, 1], c="red", s=15, label="predicted", marker="x")
    plt.legend(loc="upper right")
    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
    else:
        plt.show()


def plot_comparison_bar(results_dict, metric_name="NRMSE (%)", save_path=None):
    """
    results_dict: e.g. {"coord_shufflenet": 17.2, "heatmap_shufflenet": 14.8, ...}
    Produces a bar chart for your report figures.
    """
    labels = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="steelblue")
    plt.ylabel(metric_name)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Landmark Detection Comparison ({metric_name})")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()
    else:
        plt.show()
