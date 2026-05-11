# Federated learning with Elastic Weight Consolidation (continual tasks)

Course implementation (Category 10): **FedAvg** baseline vs **FedAvg + local EWC** on sequential tasks, using [Flower](https://flower.dev/) and PyTorch. The EWC objective follows Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks* ([arXiv:1612.00796](https://arxiv.org/abs/1612.00796)) — diagonal Fisher importance and quadratic penalty toward previous-task weights.

## Setup

Python 3.10+ recommended (tested with 3.9+).

```bash
cd fl-ewc-continual
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# installs Flower with simulation extras (Ray). For a manual install: pip install 'flwr[simulation]==...'
```

## Run an experiment

```bash
python run_experiment.py --config configs/fedavg_mnist.yaml
python run_experiment.py --config configs/ewc_mnist.yaml
```

Outputs:

- `results/<experiment_name>_metrics.csv` — round-by-round metrics (global + per-task accuracy, communication estimate).
- `results/<experiment_name>_summary.json` — CFR transitions, BWT, peak/final per-task accuracy.

Quick sanity check:

```bash
python run_experiment.py --config configs/smoke.yaml
```

## Plots (Figures 1–3)

After you have FedAvg and EWC CSVs:

```bash
python generate_plots.py --fedavg_csv results/fedavg_mnist_10clients_alpha0.1_metrics.csv \
  --ewc_csv results/ewc_mnist_10clients_alpha0.1_metrics.csv --out_dir results
```

Figures 4–7 need multiple runs (alphas, λ grid, client counts); build those tables from exported CSVs/summaries or extend plotting as needed.

## Project layout

See `FL_EWC_Implementation_Requirements.md` (course spec). This repo implements `src/` modules (`data`, `model`, `ewc`, `client`, `server`, `utils`), YAML configs under `configs/`, and `run_experiment.py`.

## Project Details

**Course Name:** Wireless Network  
**Course Code:** CS548  
**Youtube Video Link** https://youtu.be/dbJpSQ8ll4U

### Group Members
- **Harsh Kumar** (2201AI14)
- **Krrish Raj** (2201CS41)
- **Krishna Purwar** (2201CS40)
- **Rahul Nikhate** (2201CS57)
