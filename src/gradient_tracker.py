"""Per-layer gradient tracking for PyTorch training loops.

This is a standalone utility. It does not own your training loop, your model, or
your config -- you call three methods from inside whatever loop you already have:

    from src.gradient_tracker import GradientTracker

    tracker = GradientTracker(model)            # any nn.Module
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = your_loss(...)
            loss.backward()
            tracker.accumulate()                # after backward, before zero_grad
            optimizer.step()
        tracker.log_epoch(epoch, losses={"train": train_loss, "val": val_loss})
    tracker.plot_heatmap("heatmap.png")

It peels back one layer beyond the loss curve: instead of only knowing *that*
training converged, you see *where* in the network -- across depth and across
epochs -- the learning signal actually flowed.

Why a per-layer summary (the gradient L2 norm) rather than every gradient
element: storing full gradient tensors every step is infeasible for real models.
One scalar per layer keeps memory flat and answers the question that matters --
relative learning across depth and time. (Mean-absolute is also available.)

Why the norm and not a signed sum: positive and negative elements cancel in a
sum, so a hard-learning layer can read as ~0. The norm measures the magnitude of
the update signal, which is what "is this layer learning" means.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # headless-safe: works inside containers with no display
import matplotlib.pyplot as plt


class GradientTracker:
    """Capture per-layer gradient magnitude across a training run.

    Parameters
    ----------
    model : nn.Module
        Any model. Each leaf submodule that owns parameters is treated as one
        "layer", in definition order (input -> output).
    metric : str
        "l2_norm" (default) or "mean_abs".
    """

    def __init__(self, model: nn.Module, metric: str = "l2_norm"):
        if metric not in ("l2_norm", "mean_abs"):
            raise ValueError("metric must be 'l2_norm' or 'mean_abs'")
        self.model = model
        self.metric = metric

        # Group parameters by leaf module. recurse=False means only true leaves
        # (Linear, Conv, etc.) report their own params, so containers are skipped.
        self.layer_names: list[str] = []
        self._param_groups: dict[str, list[nn.Parameter]] = {}
        for mod_name, module in model.named_modules():
            own = [p for _, p in module.named_parameters(recurse=False) if p.requires_grad]
            if not own:
                continue
            label = f"{mod_name or 'root'} ({module.__class__.__name__})"
            self.layer_names.append(label)
            self._param_groups[label] = own

        if not self.layer_names:
            raise ValueError("Model has no parameter-bearing layers to track.")

        self._epoch_sum = np.zeros(len(self.layer_names), dtype=np.float64)
        self._epoch_count = 0

        self.epochs: list[int] = []
        self._history: list[np.ndarray] = []     # per-epoch per-layer values
        self.losses: dict[str, list[float]] = {}  # name -> per-epoch series

    # -- capture --------------------------------------------------------------

    def _layer_value(self, params: list[nn.Parameter]) -> float:
        if self.metric == "l2_norm":
            sq = 0.0
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach()
                    sq += float(torch.sum(g * g))
            return float(np.sqrt(sq))
        else:  # mean_abs, element-count weighted so weight & bias pool fairly
            total, count = 0.0, 0
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach()
                    total += float(torch.sum(torch.abs(g)))
                    count += g.numel()
            return total / count if count else 0.0

    def accumulate(self) -> None:
        """Record current gradients. Call once per optimizer step, after
        backward() and before the next zero_grad()."""
        for i, name in enumerate(self.layer_names):
            self._epoch_sum[i] += self._layer_value(self._param_groups[name])
        self._epoch_count += 1

    def log_epoch(self, epoch: int, losses: dict[str, float] | None = None) -> None:
        """Finalise an epoch: average the captured gradients and record any
        losses you want overlaid on the heatmap (e.g. {"train": .., "val": ..}).
        """
        if self._epoch_count == 0:
            per_layer = np.zeros(len(self.layer_names), dtype=np.float64)
        else:
            per_layer = self._epoch_sum / self._epoch_count
        self._history.append(per_layer)
        self.epochs.append(epoch)

        if losses:
            for name, value in losses.items():
                self.losses.setdefault(name, [])
                # pad if this loss appeared late, so all series stay epoch-aligned
                while len(self.losses[name]) < len(self.epochs) - 1:
                    self.losses[name].append(np.nan)
                self.losses[name].append(float(value))

        self._epoch_sum = np.zeros(len(self.layer_names), dtype=np.float64)
        self._epoch_count = 0

    # -- access ---------------------------------------------------------------

    def matrix(self) -> np.ndarray:
        """[num_layers x num_epochs] gradient-magnitude matrix."""
        if not self._history:
            raise RuntimeError("No data captured. Call accumulate()/log_epoch() during training.")
        return np.stack(self._history, axis=1)

    # -- plots ----------------------------------------------------------------

    def plot_heatmap(self, save_path: str | None = None, log_scale: bool = True):
        """Layers x epochs heatmap, with any logged losses overlaid."""
        mat = self.matrix()
        plot_mat = np.log10(mat + 1e-12) if log_scale else mat

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(plot_mat, aspect="auto", origin="lower",
                       extent=[self.epochs[0], self.epochs[-1], -0.5, len(self.layer_names) - 0.5],
                       cmap="viridis")
        ax.set_yticks(range(len(self.layer_names)))
        ax.set_yticklabels(self.layer_names, fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("layer (input → output)")
        cbar = fig.colorbar(im, ax=ax, pad=0.12)
        cbar.set_label(f"{'log10 ' if log_scale else ''}{self.metric} of gradient")

        if self.losses:
            ax2 = ax.twinx()
            for name, series in self.losses.items():
                if not np.all(np.isnan(series)):
                    ax2.plot(self.epochs, series, lw=1.6, label=f"{name} loss")
            ax2.set_yscale("log")
            ax2.set_ylabel("loss (log)")
            ax2.legend(loc="upper right", fontsize=8, framealpha=0.6)

        ax.set_title(f"Gradient flow: {self.metric} per layer across training")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig

    def plot_curves(self, save_path: str | None = None):
        """One gradient-norm trajectory per layer."""
        mat = self.matrix()
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, name in enumerate(self.layer_names):
            ax.plot(self.epochs, mat[i], label=name, lw=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{self.metric} (log)")
        ax.set_title("Per-layer gradient norm over training")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig

    def plot_contributions(self, save_path: str | None = None):
        """Bar chart: each layer's gradient magnitude integrated over the run."""
        mat = self.matrix()
        _trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.0 renamed it
        contrib = _trapz(mat, x=self.epochs, axis=1)
        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(self.layer_names))
        ax.barh(y, contrib, color="steelblue")
        ax.set_yticks(y)
        ax.set_yticklabels(self.layer_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"integrated {self.metric} over training")
        ax.set_title("Layer contribution score")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig
