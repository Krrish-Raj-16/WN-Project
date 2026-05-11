"""Elastic Weight Consolidation (Kirkpatrick et al., 2017; arXiv:1612.00796)."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_fisher(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """Diagonal Fisher ~ average squared grads of log-likelihood (per requirements)."""
    fisher: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad
    }
    model.eval()
    count = 0
    for i, (inputs, labels) in enumerate(dataloader):
        if count >= num_samples:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        output = model(inputs)
        loss = F.nll_loss(F.log_softmax(output, dim=1), labels)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        count += 1
    denom = max(count, 1)
    for n in fisher:
        fisher[n] /= denom
    return fisher


def ewc_penalty_raw(
    model: nn.Module,
    fisher_dict: Dict[int, Dict[str, torch.Tensor]],
    optpar_dict: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Sum over tasks of sum_i F_i (theta_i - theta*_i)^2 (no lambda)."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for task_id in fisher_dict:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            f = fisher_dict[task_id][n]
            opt = optpar_dict[task_id][n]
            loss = loss + (f * (p - opt).pow(2)).sum()
    return loss


class EWC:
    """Stores diagonal Fisher and snapshot parameters per consolidated task."""

    def __init__(self, lambda_ewc: float, fisher_samples: int = 100) -> None:
        self.lambda_ewc = lambda_ewc
        self.fisher_samples = fisher_samples
        self._fisher: Dict[int, Dict[str, torch.Tensor]] = {}
        self._optpar: Dict[int, Dict[str, torch.Tensor]] = {}

    def consolidate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        task_id: int,
    ) -> None:
        fisher = compute_fisher(model, dataloader, device, num_samples=self.fisher_samples)
        optpar = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._fisher[task_id] = {k: v.detach().clone() for k, v in fisher.items()}
        self._optpar[task_id] = optpar

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._fisher:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        raw = ewc_penalty_raw(model, self._fisher, self._optpar)
        return self.lambda_ewc * raw

    def tasks(self) -> Iterator[int]:
        return iter(sorted(self._fisher.keys()))

