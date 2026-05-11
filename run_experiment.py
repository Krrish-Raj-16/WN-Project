#!/usr/bin/env python3
"""Run continual federated experiments from a YAML config (Flower simulation)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import torch
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

# Ensure project root is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.client import build_client_factory
from src.data import build_all_task_train_loaders, default_data_root, get_global_testset
from src.model import get_model, get_parameters
from src.server import make_evaluate_fn, make_fit_metrics_aggregation_fn, parameters_from_model, task_schedule
from src.utils import (
    MetricsLogger,
    backward_transfer,
    catastrophic_forgetting_ratio,
    compute_comm_cost_mb,
    load_config,
    set_all_seeds,
    should_finalize_task,
    task_schedule_round_to_task,
)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_eval_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    dcfg = cfg["data"]
    dataset_name = dcfg["dataset"]
    num_classes = 100 if "100" in dataset_name.lower() else 10
    input_dim = 784 if dataset_name.lower() in ("mnist", "fmnist") else None
    mcfg = cfg.get("model", {})
    kwargs = {
        "hidden_dim": mcfg.get("hidden_dim", 400),
        "num_hidden": mcfg.get("num_hidden", 2),
        "num_classes": num_classes,
    }
    if input_dim is not None:
        kwargs["input_dim"] = input_dim
    return get_model(dataset_name, **kwargs).to(device)


def run(cfg_path: str) -> str:
    cfg = load_config(cfg_path)
    exp = cfg["experiment"]
    fed = cfg["federated"]
    lt = cfg["local_training"]
    dcfg = cfg["data"]

    seed = int(exp.get("seed", 42))
    set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = dcfg.get("data_root", default_data_root())

    num_clients = int(fed["num_clients"])
    client_fraction = float(fed["client_fraction"])
    num_tasks = int(fed["num_tasks"])
    num_rounds_per_task = int(fed["num_rounds_per_task"])
    total_rounds = num_tasks * num_rounds_per_task

    alpha = dcfg["alpha"]
    if isinstance(alpha, str) and alpha.lower() == "iid":
        alpha_arg = "iid"
    else:
        alpha_arg = float(alpha)

    task_type = dcfg.get("task_type", "permuted")
    dataset_name = dcfg["dataset"]

    train_all = build_all_task_train_loaders(
        dataset_name=dataset_name,
        task_type=task_type,
        num_clients=num_clients,
        alpha=alpha_arg,
        seed=seed,
        data_root=data_root,
        batch_size=int(lt["batch_size"]),
        num_tasks=num_tasks,
    )

    test_loaders: Dict[int, torch.utils.data.DataLoader] = {}
    for t in range(num_tasks):
        test_loaders[t] = get_global_testset(
            dataset_name,
            t,
            task_type,
            num_tasks,
            data_root,
            permutation_seed=10_000 + t,
            batch_size=256,
        )

    eval_model = _make_eval_model(cfg, device)
    num_params = _count_parameters(eval_model)

    def model_ctor() -> torch.nn.Module:
        m = _make_eval_model(cfg, device)
        return m

    use_ewc = bool(cfg.get("ewc", {}).get("enabled", False))
    client_fn = build_client_factory(cfg, train_all, test_loaders, model_ctor, device, use_ewc)

    exp_name = exp["name"]
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{exp_name}_metrics.csv")

    num_task_cols = num_tasks
    cols = (
        ["round", "task_id", "train_loss", "global_test_acc", "global_test_loss", "comm_cost_mb"]
        + [f"task_{i}_acc" for i in range(num_tasks)]
    )
    metrics_logger = MetricsLogger(csv_path, columns=cols)
    metrics_state: Dict[str, float] = {"train_loss": float("nan")}

    evaluate_fn = make_evaluate_fn(
        eval_model,
        test_loaders,
        device,
        metrics_logger,
        metrics_state,
        num_rounds_per_task,
        num_params,
        client_fraction,
        num_clients,
    )

    def on_fit_config_fn(server_round: int) -> Dict[str, Any]:
        tid = task_schedule_round_to_task(server_round, num_rounds_per_task, num_tasks)
        finalize = should_finalize_task(server_round, num_rounds_per_task, total_rounds)
        return {
            "task_id": tid,
            "finalize_task": finalize,
            "local_epochs": int(lt["epochs"]),
            "lr": float(lt["lr"]),
        }

    initial_model = _make_eval_model(cfg, device)
    strategy = FedAvg(
        fraction_fit=client_fraction,
        fraction_evaluate=0.0,
        min_fit_clients=max(2, int(round(client_fraction * num_clients))),
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=parameters_from_model(initial_model),
        fit_metrics_aggregation_fn=make_fit_metrics_aggregation_fn(metrics_state),
    )

    _ = task_schedule(total_rounds, num_tasks, num_rounds_per_task)

    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    summary_path = os.path.join(results_dir, f"{exp_name}_summary.json")
    peak_accs = _peak_task_accs(csv_path, num_tasks, num_rounds_per_task)
    final_accs = _final_task_accs(csv_path, num_tasks)
    cfr = _compute_cfr_transitions(csv_path, num_tasks, num_rounds_per_task)

    summary = {
        "experiment": exp_name,
        "config_path": cfg_path,
        "total_round_comm_mb": compute_comm_cost_mb(
            num_params, client_fraction, num_clients, total_rounds
        ),
        "peak_task_acc": peak_accs,
        "final_task_acc": final_accs,
        "cfr": cfr,
        "bwt": backward_transfer(peak_accs, final_accs),
    }

    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path


def _read_metrics(csv_path: str):
    import pandas as pd

    return pd.read_csv(csv_path)


def _peak_task_accs(csv_path: str, num_tasks: int, num_rounds_per_task: int):
    """Per-task accuracy right after that task's segment ends (round k * R)."""
    df = _read_metrics(csv_path)
    peaks = []
    for t in range(num_tasks):
        end_r = (t + 1) * num_rounds_per_task
        row = df.loc[df["round"] == end_r].iloc[-1]
        peaks.append(float(row[f"task_{t}_acc"]))
    return peaks


def _final_task_accs(csv_path: str, num_tasks: int):
    df = _read_metrics(csv_path)
    last = df.iloc[-1]
    return [float(last[f"task_{i}_acc"]) for i in range(num_tasks)]


def _compute_cfr_transitions(csv_path: str, num_tasks: int, num_rounds_per_task: int):
    """CFR after training next task: compare task-t accuracy at end of task t vs end of task t+1."""
    df = _read_metrics(csv_path)
    out: Dict[str, float] = {}
    if num_tasks < 2:
        return out
    for t in range(num_tasks - 1):
        end_a = (t + 1) * num_rounds_per_task
        end_b = (t + 2) * num_rounds_per_task
        row_a = df.loc[df["round"] == end_a].iloc[-1]
        row_b = df.loc[df["round"] == end_b].iloc[-1]
        key = f"task_{t}_acc"
        out[f"cfr_{t}_to_{t+1}"] = catastrophic_forgetting_ratio(
            float(row_a[key]),
            float(row_b[key]),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual federated learning (FedAvg / EWC)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    out = run(args.config)
    print(f"Finished. Metrics written to {out}")


if __name__ == "__main__":
    main()
