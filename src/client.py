"""Flower NumPy clients: FedAvg baseline and FedAvg + local EWC."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Config, NDArrays, Scalar

from .ewc import EWC
from .model import get_parameters, set_parameters


class FedAvgClient(fl.client.NumPyClient):
    """Standard local SGD with optional task switching via fit config."""

    def __init__(
        self,
        cid: int,
        model: nn.Module,
        train_loaders_by_task: List[torch.utils.data.DataLoader],
        test_loaders_by_task: List[torch.utils.data.DataLoader],
        device: torch.device,
        momentum: float = 0.9,
    ) -> None:
        self.cid = cid
        self.model = model
        self.train_loaders_by_task = train_loaders_by_task
        self.test_loaders_by_task = test_loaders_by_task
        self.device = device
        self.momentum = momentum

    def get_parameters(self, config: Config) -> NDArrays:
        return get_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        set_parameters(self.model, parameters)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        task_id = int(config["task_id"])
        epochs = int(config["local_epochs"])
        lr = float(config["lr"])
        finalize_task = bool(config.get("finalize_task", False))

        loader = self.train_loaders_by_task[task_id]
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.momentum,
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics: Dict[str, Scalar] = {"train_loss": avg_loss}
        return self.get_parameters(config), len(loader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        task_id = int(config.get("task_id", 0))
        loader = self.test_loaders_by_task[task_id]
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss_sum += float(criterion(logits, y).item()) * y.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += y.size(0)
        return loss_sum / max(total, 1), total, {"accuracy": correct / max(total, 1)}


class EWCClient(FedAvgClient):
    """Local training with EWC penalty; consolidates Fisher at task boundaries."""

    def __init__(
        self,
        cid: int,
        model: nn.Module,
        train_loaders_by_task: List[torch.utils.data.DataLoader],
        test_loaders_by_task: List[torch.utils.data.DataLoader],
        device: torch.device,
        ewc_lambda: float,
        fisher_samples: int,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(
            cid=cid,
            model=model,
            train_loaders_by_task=train_loaders_by_task,
            test_loaders_by_task=test_loaders_by_task,
            device=device,
            momentum=momentum,
        )
        self.ewc = EWC(lambda_ewc=ewc_lambda, fisher_samples=fisher_samples)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        task_id = int(config["task_id"])
        epochs = int(config["local_epochs"])
        lr = float(config["lr"])
        finalize_task = bool(config.get("finalize_task", False))

        loader = self.train_loaders_by_task[task_id]
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.momentum,
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y) + self.ewc.penalty(self.model)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

        if finalize_task:
            self.ewc.consolidate(self.model, loader, self.device, task_id)

        avg_loss = total_loss / max(n_batches, 1)
        return self.get_parameters(config), len(loader.dataset), {"train_loss": avg_loss}


def build_client_factory(
    cfg: Dict,
    train_all_tasks: List[List[torch.utils.data.DataLoader]],
    test_loaders_by_task: Dict[int, torch.utils.data.DataLoader],
    model_ctor: Callable[[], nn.Module],
    device: torch.device,
    use_ewc: bool,
) -> Callable[[str], fl.client.Client]:
    """Returns Flower client_fn(cid_str)."""

    num_clients = len(train_all_tasks[0])

    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        train_by_task = [train_all_tasks[t][cid_int] for t in range(len(train_all_tasks))]
        test_by_task = [test_loaders_by_task[t] for t in sorted(test_loaders_by_task.keys())]
        model = model_ctor()
        lt = cfg["local_training"]
        if use_ewc:
            ecfg = cfg.get("ewc", {})
            return EWCClient(
                cid=cid_int,
                model=model,
                train_loaders_by_task=train_by_task,
                test_loaders_by_task=test_by_task,
                device=device,
                ewc_lambda=float(ecfg.get("lambda", 400)),
                fisher_samples=int(ecfg.get("fisher_samples", 100)),
                momentum=float(lt.get("momentum", 0.9)),
            ).to_client()
        return FedAvgClient(
            cid=cid_int,
            model=model,
            train_loaders_by_task=train_by_task,
            test_loaders_by_task=test_by_task,
            device=device,
            momentum=float(lt.get("momentum", 0.9)),
        ).to_client()

    _ = num_clients  # silence lint if unused
    return client_fn

