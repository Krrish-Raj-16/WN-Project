"""Seeding, communication cost, YAML config, metrics logging, CFR/BWT, plots."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import rcParams


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_comm_cost_mb(
    num_parameters: int,
    client_fraction: float,
    num_clients: int,
    num_rounds: int,
    bytes_per_param: int = 4,
) -> float:
    """Upper-bound MB uploaded+downloaded per round estimate (symmetric FedAvg)."""
    clients_per_round = max(1, int(round(client_fraction * num_clients)))
    bytes_total = 2 * num_parameters * bytes_per_param * clients_per_round * num_rounds
    return bytes_total / (1024**2)


class MetricsLogger:
    """Append per-round metrics to CSV."""

    def __init__(self, path: str, columns: Optional[List[str]] = None) -> None:
        self.path = path
        self.columns = columns
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._written_header = False

    def log(self, row: Dict[str, Any]) -> None:
        df = pd.DataFrame([row])
        if not self._written_header:
            cols = self.columns or list(row.keys())
            df = df.reindex(columns=cols)
            df.to_csv(self.path, mode="w", index=False)
            self._written_header = True
            self.columns = cols
        else:
            df.to_csv(self.path, mode="a", header=False, index=False)


def task_schedule_round_to_task(
    server_round: int,
    num_rounds_per_task: int,
    num_tasks: int,
) -> int:
    """Flower round index → current task id (round 0 = pre-training eval → task 0)."""
    if server_round <= 0:
        return 0
    idx = (server_round - 1) // num_rounds_per_task
    return int(min(max(idx, 0), num_tasks - 1))


def should_finalize_task(
    server_round: int,
    num_rounds_per_task: int,
    total_rounds: int,
) -> bool:
    """True after the last local training step of a task segment."""
    if server_round > total_rounds:
        return False
    return server_round % num_rounds_per_task == 0


def catastrophic_forgetting_ratio(
    acc_after_task_a_peak: float,
    acc_task_a_after_next: float,
) -> float:
    """CFR as in requirements (avoid div by zero)."""
    if acc_after_task_a_peak <= 1e-8:
        return 0.0
    return (acc_after_task_a_peak - acc_task_a_after_next) / acc_after_task_a_peak


def backward_transfer(
    peak_accs: List[float],
    final_accs: List[float],
) -> float:
    """BWT: average (final_i - peak_i) for i < T-1."""
    if len(peak_accs) < 2:
        return 0.0
    t = len(peak_accs)
    vals = []
    for i in range(t - 1):
        vals.append(final_accs[i] - peak_accs[i])
    return float(np.mean(vals))


def plot_learning_curves(
    out_dir: str,
    fedavg_csv: str,
    ewc_csv: str,
    dpi: int = 300,
) -> None:
    """Figure 1–3 style plots from two metrics CSV paths."""
    rcParams.update({"font.size": 11})
    fa = pd.read_csv(fedavg_csv)
    ew = pd.read_csv(ewc_csv)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.plot(fa["round"], fa["global_test_acc"], label="FedAvg")
    ax.plot(ew["round"], ew["global_test_acc"], label="FedAvg+EWC")
    ax.set_xlabel("Round")
    ax.set_ylabel("Global test accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure1_global_accuracy.png"))
    fig.savefig(os.path.join(out_dir, "figure1_global_accuracy.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.plot(fa["round"], fa["global_test_loss"], label="FedAvg")
    ax.plot(ew["round"], ew["global_test_loss"], label="FedAvg+EWC")
    ax.set_xlabel("Round")
    ax.set_ylabel("Global test loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure2_global_loss.png"))
    fig.savefig(os.path.join(out_dir, "figure2_global_loss.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    task_cols = [c for c in fa.columns if c.endswith("_acc") and c.startswith("task_")]
    for c in task_cols:
        ax.plot(fa["round"], fa[c], linestyle="--", alpha=0.7, label=f"FedAvg {c}")
    for c in task_cols:
        if c in ew.columns:
            ax.plot(ew["round"], ew[c], label=f"EWC {c}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Task-specific accuracy")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure3_task_accuracy.png"))
    fig.savefig(os.path.join(out_dir, "figure3_task_accuracy.pdf"))
    plt.close(fig)


def save_summary_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

