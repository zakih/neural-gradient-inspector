"""Train the damped-oscillator PINN from the command line.

Usage:
    python train.py                                  # defaults from configs/default.yaml
    python train.py training.epochs=500 physics.c=0.1
    python train.py --config configs/default.yaml visualization.gradient_metric=mean_abs

This mirrors notebooks/training.ipynb for users who prefer a script. Plots and
the MLflow run land under runs/.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from src.config import Config
from src.dataset import OscillatorDataset, make_dataloaders
from src.model import MLP
from src.trainer import Trainer
from visualizations import GradientTracker, plot_architecture

# MLflow is optional: the framework runs without it, just without tracking.
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. training.epochs=500")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.overrides:
        cfg = cfg.apply_overrides(args.overrides)

    os.makedirs("runs", exist_ok=True)

    dataset = OscillatorDataset(cfg.physics)
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset, batch_size=256, val_split=0.15, test_split=0.15,
        num_workers=0, seed=cfg.experiment.seed,
    )

    model = MLP(cfg.model)
    print(f"Model: MLP, {model.num_parameters()} trainable parameters")

    # Architecture diagram (Mermaid source saved for the README / notebook).
    mermaid = plot_architecture(model, input_shape=(1, cfg.model.in_dim),
                                save_path="runs/architecture.mmd")
    print("Architecture diagram written to runs/architecture.mmd")

    tracker = GradientTracker(model, metric=cfg.visualization.gradient_metric)

    if mlflow is not None:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_ctx = mlflow.start_run(run_name=cfg.experiment.name) if mlflow is not None else _Null()
    with run_ctx:
        if mlflow is not None:
            mlflow.log_params(cfg.to_flat_dict())
            cfg.save_yaml("runs/config_snapshot.yaml")
            mlflow.log_artifact("runs/config_snapshot.yaml")
            mlflow.log_artifact("runs/architecture.mmd")

        trainer = Trainer(model, dataset, cfg, tracker=tracker, mlflow_module=mlflow)
        result = trainer.fit(train_loader, val_loader, test_loader, run_dir="runs")
        print(f"\nBest metric {result['best_metric']:.4e} at epoch {result['best_epoch']} "
              f"in {result['duration_s']:.1f}s")
        if "test" in result:
            print(f"Test loss: {result['test']['total']:.4e}")

        # Gradient-flow plots -- the headline artifacts.
        if cfg.visualization.save_all_plots:
            tracker.plot_heatmap("runs/gradient_heatmap.png")
            tracker.plot_curves("runs/gradient_curves.png")
            tracker.plot_contributions("runs/layer_contributions.png")
            print("Gradient plots written to runs/")
            if mlflow is not None:
                for p in ["gradient_heatmap.png", "gradient_curves.png", "layer_contributions.png"]:
                    mlflow.log_artifact(f"runs/{p}")

        # Prediction vs analytical ground truth.
        _plot_solution(model, dataset, cfg, save_path="runs/solution_vs_analytical.png")
        if mlflow is not None:
            mlflow.log_artifact("runs/solution_vs_analytical.png")

    print("Done.")
    return result


def _plot_solution(model, dataset, cfg, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    model.eval()
    t = np.linspace(0.0, cfg.physics.t_max, 500, dtype=np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(t).unsqueeze(1)).squeeze(1).cpu().numpy()
    exact = dataset.analytical_solution(t)
    mae = float(np.mean(np.abs(pred - exact)))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, exact, "k-", lw=2, label="analytical")
    ax.plot(t, pred, "r--", lw=1.6, label="PINN")
    ax.set_xlabel("t"); ax.set_ylabel("x(t)")
    ax.set_title(f"Damped oscillator: PINN vs analytical (MAE={mae:.3e})")
    ax.legend(); ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(save_path, dpi=130)
    print(f"Solution comparison written to {save_path} (MAE={mae:.3e})")


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
