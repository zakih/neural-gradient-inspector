"""Core training loop with MLflow integration and gradient-flow tracking.

The loss for the damped oscillator PINN has two competing terms:

  L = w_physics * || m x'' + c x' + k x ||^2   (PDE residual, on collocation pts)
    + w_ic      * [ (x(0)-x0)^2 + (x'(0)-v0)^2 ]  (initial conditions)

These two terms typically produce gradients of very different magnitudes -- the
canonical PINN training pathology. The GradientTracker is wired in precisely to
make that imbalance visible across layers and epochs.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch

from .utils import set_seed, get_device, device_info, save_checkpoint


def _grad(outputs, inputs):
    """First derivative d(outputs)/d(inputs) via autograd, graph retained."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
    )[0]


def oscillator_residual_loss(model, t, physics):
    """PDE residual mean-squared error at collocation points t (requires grad)."""
    t = t.clone().requires_grad_(True)
    x = model(t)
    x_t = _grad(x, t)
    x_tt = _grad(x_t, t)
    residual = physics.m * x_tt + physics.c * x_t + physics.k * x
    return torch.mean(residual ** 2)


def initial_condition_loss(model, physics, device):
    """Enforce x(0)=x0 and x'(0)=v0."""
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)
    x0_pred = model(t0)
    v0_pred = _grad(x0_pred, t0)
    loss_x = (x0_pred - physics.x0) ** 2
    loss_v = (v0_pred - physics.v0) ** 2
    return torch.mean(loss_x) + torch.mean(loss_v)


def compute_loss(model, t_batch, physics, training_cfg, device):
    """Weighted composite PINN loss. Returns (total, physics_term, ic_term)."""
    phys = oscillator_residual_loss(model, t_batch, physics)
    ic = initial_condition_loss(model, physics, device)
    total = training_cfg.w_physics * phys + training_cfg.w_ic * ic
    return total, phys.detach(), ic.detach()


def _make_scheduler(optimizer, training_cfg):
    name = (training_cfg.scheduler or "none").lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)
    raise ValueError(f"Unknown scheduler '{training_cfg.scheduler}'")


class Trainer:
    """Owns the training loop, logging, checkpointing, and gradient tracking."""

    def __init__(self, model, dataset, config, tracker=None, mlflow_module=None):
        self.cfg = config
        self.model = model
        self.dataset = dataset
        self.tracker = tracker
        self.mlflow = mlflow_module  # injected so the framework runs without mlflow installed

        set_seed(config.experiment.seed)
        self.device = get_device()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = _make_scheduler(self.optimizer, config.training)

    # -- one epoch ------------------------------------------------------------

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        totals = {"total": 0.0, "physics": 0.0, "ic": 0.0}
        n_batches = 0
        for t_batch in loader:
            t_batch = t_batch.to(self.device)
            if train:
                self.optimizer.zero_grad()
            # residual needs grad even at eval time (it differentiates the net),
            # so we do NOT wrap the forward in torch.no_grad().
            loss, phys, ic = compute_loss(self.model, t_batch, self.cfg.physics,
                                          self.cfg.training, self.device)
            if train:
                loss.backward()
                if self.cfg.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.gradient_clip_norm)
                if self.tracker is not None:
                    self.tracker.accumulate()  # read .grad before step
                self.optimizer.step()
            totals["total"] += float(loss.detach())
            totals["physics"] += float(phys)
            totals["ic"] += float(ic)
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # -- full training --------------------------------------------------------

    def fit(self, train_loader, val_loader=None, test_loader=None, run_dir="runs"):
        os.makedirs(run_dir, exist_ok=True)
        best_metric = float("inf") if self.cfg.checkpointing.mode == "min" else -float("inf")
        best_epoch = -1
        epochs_since_improve = 0
        start = time.time()

        for epoch in range(self.cfg.training.epochs):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False) if val_loader else {}

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("total", train_metrics["total"]))
                else:
                    self.scheduler.step()

            val_total = val_metrics.get("total", np.nan) if val_metrics else None
            if self.tracker is not None and (epoch % self.cfg.visualization.gradient_capture_interval == 0):
                self.tracker.on_epoch_end(epoch, train_metrics["total"], val_total)

            self._log_epoch(epoch, train_metrics, val_metrics)

            # checkpoint / early stopping on the configured metric
            metric_val = (val_metrics.get("total") if val_metrics else train_metrics["total"])
            improved = (metric_val < best_metric if self.cfg.checkpointing.mode == "min"
                        else metric_val > best_metric)
            if improved:
                best_metric, best_epoch = metric_val, epoch
                epochs_since_improve = 0
                if self.cfg.checkpointing.save_best:
                    save_checkpoint(os.path.join("checkpoints", "best.pt"),
                                    self.model, self.optimizer, epoch, best_metric,
                                    self.cfg.to_dict())
            else:
                epochs_since_improve += 1

            if (self.cfg.checkpointing.save_every_n_epochs > 0
                    and epoch % self.cfg.checkpointing.save_every_n_epochs == 0):
                save_checkpoint(os.path.join("checkpoints", "latest.pt"),
                                self.model, self.optimizer, epoch, best_metric,
                                self.cfg.to_dict())

            if (self.cfg.training.early_stopping_patience > 0
                    and epochs_since_improve >= self.cfg.training.early_stopping_patience):
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{epochs_since_improve} epochs)")
                break

        duration = time.time() - start
        if self.mlflow is not None:
            self.mlflow.log_metric("best_metric", best_metric)
            self.mlflow.log_metric("best_epoch", best_epoch)
            self.mlflow.log_metric("training_seconds", duration)

        result = {"best_metric": best_metric, "best_epoch": best_epoch,
                  "duration_s": duration}
        if test_loader is not None:
            test_metrics = self._run_epoch(test_loader, train=False)
            result["test"] = test_metrics
            if self.mlflow is not None:
                for k, v in test_metrics.items():
                    self.mlflow.log_metric(f"test_{k}", v)
        return result

    def _log_epoch(self, epoch, train_metrics, val_metrics):
        lr = self.optimizer.param_groups[0]["lr"]
        if self.mlflow is not None:
            self.mlflow.log_metric("lr", lr, step=epoch)
            for k, v in train_metrics.items():
                self.mlflow.log_metric(f"train_{k}", v, step=epoch)
            for k, v in val_metrics.items():
                self.mlflow.log_metric(f"val_{k}", v, step=epoch)
        if epoch % 25 == 0 or epoch == self.cfg.training.epochs - 1:
            vs = f" | val {val_metrics.get('total', float('nan')):.3e}" if val_metrics else ""
            print(f"epoch {epoch:4d} | train {train_metrics['total']:.3e} "
                  f"(phys {train_metrics['physics']:.2e}, ic {train_metrics['ic']:.2e}){vs} | lr {lr:.2e}")
