#!/usr/bin/env python3
"""Generate comparison plots after metrics CSVs exist (Figures 1–3 from utils.plot_learning_curves)."""

from __future__ import annotations

import argparse
import os

from src.utils import plot_learning_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fedavg_csv",
        type=str,
        required=True,
        help="Path to FedAvg metrics CSV (e.g. results/fedavg_mnist_*_metrics.csv)",
    )
    parser.add_argument(
        "--ewc_csv",
        type=str,
        required=True,
        help="Path to FedAvg+EWC metrics CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory for PNG/PDF figures",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_learning_curves(args.out_dir, args.fedavg_csv, args.ewc_csv)


if __name__ == "__main__":
    main()
