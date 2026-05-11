"""Dataset loading, permuted tasks, Dirichlet partitioning (non-IID)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_permutation(seed: int, size: int = 784) -> np.ndarray:
    """Fixed pixel permutation index map for permuted-MNIST-style tasks."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(size).astype(np.int64)
    return perm


class PermutedDataset(Dataset):
    """Wraps a vision dataset and applies a fixed flat permutation to pixels."""

    def __init__(self, base: Dataset, perm: np.ndarray, flatten: bool = True) -> None:
        self.base = base
        self.perm = perm
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if isinstance(x, torch.Tensor):
            arr = x.view(-1).numpy().astype(np.float32)
        else:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
        arr = arr[self.perm]
        if self.flatten:
            x_out = torch.from_numpy(arr)
        else:
            side = int(np.sqrt(arr.shape[0]))
            x_out = torch.from_numpy(arr).view(1, side, side)
        return x_out, y


def _split_task_classes(num_tasks: int, total_classes: int) -> List[List[int]]:
    """Disjoint class groups for split-task continual learning (even split)."""
    classes = list(range(total_classes))
    base = total_classes // num_tasks
    rem = total_classes % num_tasks
    groups: List[List[int]] = []
    start = 0
    for i in range(num_tasks):
        length = base + (1 if i < rem else 0)
        groups.append(classes[start : start + length])
        start += length
    return groups


def _subset_by_classes(dataset: Dataset, classes: Sequence[int]) -> Subset:
    cls_set = set(classes)
    indices = [i for i, (_, y) in enumerate(dataset) if int(y) in cls_set]
    return Subset(dataset, indices)


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: Union[float, str],
    seed: int,
) -> List[Subset]:
    """Dirichlet label skew (non-IID); alpha='iid' or large homogeneous split."""
    rng = np.random.RandomState(seed)
    labels = np.array([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)
    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    if isinstance(alpha, str) and alpha.lower() == "iid":
        order = rng.permutation(len(dataset))
        splits = np.array_split(order, num_clients)
        return [Subset(dataset, s.tolist()) for s in splits]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        rng.shuffle(idx_k)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return [Subset(dataset, sorted(idxs)) for idxs in client_indices]


def _mnist_family(
    name: str,
    train: bool,
    data_root: str,
) -> Dataset:
    tfm = transforms.Compose([transforms.ToTensor()])
    if name == "mnist":
        return datasets.MNIST(
            root=data_root, train=train, download=True, transform=tfm
        )
    if name == "fmnist":
        return datasets.FashionMNIST(
            root=data_root, train=train, download=True, transform=tfm
        )
    raise ValueError(name)


def _cifar(name: str, train: bool, data_root: str) -> Dataset:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    if name in ("cifar10", "cifar-10"):
        return datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=tfm
        )
    if name in ("cifar100", "cifar-100"):
        return datasets.CIFAR100(
            root=data_root, train=train, download=True, transform=tfm
        )
    raise ValueError(name)


def load_dataset(
    name: str,
    task_id: int,
    task_type: str,
    train: bool,
    data_root: str,
    permutation_seed: Optional[int] = None,
    num_tasks: int = 3,
) -> Dataset:
    """
    Load train or test split for a dataset/task.
    task_type: 'permuted' | 'split'
    """
    name_l = name.lower()
    if name_l in ("mnist", "fmnist"):
        base = _mnist_family(name_l, train, data_root)
        if task_type == "permuted":
            seed = (
                permutation_seed
                if permutation_seed is not None
                else 10_000 + int(task_id)
            )
            perm = get_permutation(seed, 784)
            return PermutedDataset(base, perm, flatten=True)
        if task_type == "split":
            groups = _split_task_classes(num_tasks, 10)
            return _subset_by_classes(base, groups[task_id])
        raise ValueError(task_type)

    if name_l in ("cifar10", "cifar-10", "cifar100", "cifar-100"):
        base = _cifar(name_l, train, data_root)
        num_classes = 100 if "100" in name_l else 10
        if task_type == "split":
            groups = _split_task_classes(num_tasks, num_classes)
            return _subset_by_classes(base, groups[task_id])
        if task_type == "permuted":
            raise ValueError("Permuted task mode is defined for MNIST/FMNIST in this repo.")
        raise ValueError(task_type)

    raise ValueError(f"Unknown dataset {name}")


def get_global_testset(
    dataset_name: str,
    task_id: int,
    task_type: str,
    num_tasks: int,
    data_root: str,
    permutation_seed: Optional[int] = None,
    batch_size: int = 128,
) -> DataLoader:
    """Centralized test loader for server-side evaluation."""
    ds = load_dataset(
        dataset_name,
        task_id,
        task_type,
        train=False,
        data_root=data_root,
        permutation_seed=permutation_seed,
        num_tasks=num_tasks,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_task(
    dataset_name: str,
    task_id: int,
    task_type: str,
    num_clients: int,
    alpha: Union[float, str],
    seed: int,
    data_root: str,
    batch_size: int,
    num_tasks: int,
) -> Tuple[List[DataLoader], Dict[int, DataLoader]]:
    """
    Build per-client train loaders and per-task global test loaders for all tasks 0..T-1.
    Returns (client_train_loaders_for_this_task, global_test_loaders_by_task_id).
    """
    train_full = load_dataset(
        dataset_name,
        task_id,
        task_type,
        train=True,
        data_root=data_root,
        permutation_seed=10_000 + task_id,
        num_tasks=num_tasks,
    )
    parts = dirichlet_partition(train_full, num_clients, alpha, seed + task_id)
    train_loaders = [
        DataLoader(p, batch_size=batch_size, shuffle=True, drop_last=False)
        for p in parts
    ]

    test_by_task: Dict[int, DataLoader] = {}
    for t in range(num_tasks):
        test_by_task[t] = get_global_testset(
            dataset_name,
            t,
            task_type,
            num_tasks,
            data_root,
            permutation_seed=10_000 + t,
            batch_size=batch_size,
        )
    return train_loaders, test_by_task


def build_all_task_train_loaders(
    dataset_name: str,
    task_type: str,
    num_clients: int,
    alpha: Union[float, str],
    seed: int,
    data_root: str,
    batch_size: int,
    num_tasks: int,
) -> List[List[DataLoader]]:
    """For each task_id, list of client train loaders."""
    all_loaders: List[List[DataLoader]] = []
    for task_id in range(num_tasks):
        train_full = load_dataset(
            dataset_name,
            task_id,
            task_type,
            train=True,
            data_root=data_root,
            permutation_seed=10_000 + task_id,
            num_tasks=num_tasks,
        )
        parts = dirichlet_partition(train_full, num_clients, alpha, seed + task_id)
        all_loaders.append(
            [
                DataLoader(p, batch_size=batch_size, shuffle=True, drop_last=False)
                for p in parts
            ]
        )
    return all_loaders


def default_data_root() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def apply_permutation(dataset: Dataset, perm: np.ndarray, flatten: bool = True) -> PermutedDataset:
    """Wrap `dataset` so inputs use the given pixel permutation."""
    return PermutedDataset(dataset, perm, flatten=flatten)


partition_data = dirichlet_partition
