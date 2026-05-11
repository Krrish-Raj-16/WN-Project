"""Continual federated strategy: task schedule + centralized evaluation + CSV logging."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.common import Metrics, NDArrays, Parameters, Scalar

from .model import get_parameters, set_parameters
from .utils import MetricsLogger, task_schedule_round_to_task


def task_schedule(total_rounds: int, num_tasks: int, num_rounds_per_task: int) -> Dict[int, int]:
    """Map 1-based round index → task id (for inspection / debugging)."""
    out: Dict[int, int] = {}
    for r in range(1, total_rounds + 1):
        out[r] = task_schedule_round_to_task(r, num_rounds_per_task, num_tasks)
    return out


def _eval_loader(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def make_evaluate_fn(
    model: nn.Module,
    test_loaders_by_task: Dict[int, torch.utils.data.DataLoader],
    device: torch.device,
    metrics_logger: MetricsLogger,
    metrics_state: Dict[str, float],
    num_rounds_per_task: int,
    num_parameters: int,
    client_fraction: float,
    num_clients: int,
) -> Callable[..., Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Server-side evaluation on global test sets for each task."""

    num_tasks = len(test_loaders_by_task)

    def evaluate(
        server_round: int,
        parameters_ndarrays: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_parameters(model, parameters_ndarrays)

        task_losses: List[float] = []
        task_accs: List[float] = []
        for t in range(num_tasks):
            loss_t, acc_t = _eval_loader(model, test_loaders_by_task[t], device)
            task_losses.append(loss_t)
            task_accs.append(acc_t)

        global_loss = float(np.mean(task_losses))
        global_acc = float(np.mean(task_accs))

        tid = task_schedule_round_to_task(server_round, num_rounds_per_task, num_tasks)

        comm_mb = (
            2
            * num_parameters
            * 4
            * max(1, int(round(client_fraction * num_clients)))
            * server_round
            / (1024**2)
        )

        row = {
            "round": server_round,
            "task_id": tid,
            "train_loss": metrics_state.get("train_loss", float("nan")),
            "global_test_acc": global_acc,
            "global_test_loss": global_loss,
            "comm_cost_mb": comm_mb,
        }
        for i in range(num_tasks):
            row[f"task_{i}_acc"] = task_accs[i]
        metrics_logger.log(row)

        metrics_out: Dict[str, Scalar] = {
            "accuracy": global_acc,
            "global_test_loss": global_loss,
        }
        for i in range(num_tasks):
            metrics_out[f"task_{i}_acc"] = task_accs[i]
        return global_loss, metrics_out

    return evaluate


def make_fit_metrics_aggregation_fn(
    metrics_state: Dict[str, float],
) -> Callable[[List[Tuple[int, Metrics]]], Metrics]:
    """Weighted train_loss across clients; mirrors Flower FedAvg weighting."""

    def aggregate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        weighted = 0.0
        denom = 0
        for num_examples, m in metrics:
            if m and "train_loss" in m:
                weighted += float(m["train_loss"]) * num_examples
                denom += num_examples
        tl = weighted / denom if denom > 0 else float("nan")
        metrics_state["train_loss"] = tl
        return {"train_loss": tl}

    return aggregate


def parameters_from_model(model: nn.Module) -> Parameters:
    from flwr.common import ndarrays_to_parameters

    return ndarrays_to_parameters(get_parameters(model))

