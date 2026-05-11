"""Neural network architectures and Flower-compatible parameter helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import NDArrays


class MLP(nn.Module):
    """Fully-connected network with ReLU (paper Appendix 4.1 style)."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        output_dim: int = 10,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10 (32x32 RGB)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_model(dataset_name: str, **kwargs) -> nn.Module:
    """Factory for dataset-appropriate models."""
    name = dataset_name.lower()
    if name in ("mnist", "fmnist"):
        return MLP(
            input_dim=kwargs.get("input_dim", 784),
            hidden_dim=kwargs.get("hidden_dim", 400),
            output_dim=kwargs.get("num_classes", 10),
            num_hidden_layers=kwargs.get("num_hidden", 2),
        )
    if name in ("cifar10", "cifar-10"):
        return SmallCNN(num_classes=kwargs.get("num_classes", 10))
    if name in ("cifar100", "cifar-100"):
        return SmallCNN(num_classes=kwargs.get("num_classes", 100))
    raise ValueError(f"Unsupported dataset for model factory: {dataset_name}")


def get_parameters(model: nn.Module) -> NDArrays:
    """Export model weights as NumPy arrays for Flower."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    """Load Flower NumPy arrays into model.state_dict()."""
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state_dict, strict=True)

