"""Gradient flow analysis -- the inspection component.

Core question: as training proceeds, WHERE in the network is learning actually
happening, and does it stay healthy?

`GradientTracker` records one scalar per layer per capture -- the L2 norm of that
layer's gradient (mean-absolute is also available) -- building a compact
[num_layers x num_epochs] matrix that drives three views:

  * heatmap        : layers (y) x epochs (x), colour = gradient magnitude, with
                     train/val loss overlaid on a twin axis
  * curves         : per-layer gradient-norm trajectories over epochs
  * contributions  : per-layer gradient magnitude integrated over the whole run

Why a per-layer summary rather than per-element: storing every gradient element
for every step is infeasible for real models. A single norm per layer keeps
memory flat (num_layers floats per capture) and answers the question that
matters -- relative learning across depth and time.

Why L2 norm rather than a signed sum: positive and negative gradient elements
cancel in a sum, so a hard-learning layer can read as ~0. The norm measures
magnitude of the update signal, which is what "is this layer learning" means.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # headless: works inside the container with no display
import matplotlib.pyplot as plt


class GradientTracker:
    """Capture per-layer gradient magnitude across training.

    Typical use:

        tracker = GradientTracker(model, metric="l2_norm")
        for epoch in range(E):
            for batch in loader:
                optimizer.zero_grad()
                loss = compute_loss(...)
                loss.backward()
                tracker.accumulate()      # read .grad before step() clears nothing,
                optimizer.step()          # but call before zero_grad of next iter
            tracker.on_epoch_end(epoch, train_loss, val_loss)
        tracker.plot_heatmap("runs/gradient_heatmap.png")

    Note: this reads parameter .grad directly, so there are no forward/backward
    hooks to leak or detach. `attach()` is kept as a no-op alias for API
    familiarity but is not required.
    """

    def __init__(self, model: nn.Module, metric: str = "l2_norm"):
        if metric not in ("l2_norm", "mean_abs"):
            raise ValueError("metric must be 'l2_norm' or 'mean_abs'")
        self.model = model
        self.metric = metric

        # Group parameters by "layer". We treat each leaf module that owns
        # parameters as one layer, preserving definition order (input -> output).
        self.layer_names: list[str] = []
        self._param_groups: dict[str, list[torch.nn.Parameter]] = {}
        for mod_name, module in model.named_modules():
            own = [p for n, p in module.named_parameters(recurse=False) if p.requires_grad]
            if not own:
                continue
            # Skip container modules; only leaves own parameters with recurse=False,
            # so this naturally selects Linear/Conv/etc.
            label = f"{mod_name or 'root'} ({module.__class__.__name__})"
            self.layer_names.append(label)
            self._param_groups[label] = own

        # within-epoch accumulation
        self._epoch_sum = np.zeros(len(self.layer_names), dtype=np.float64)
        self._epoch_count = 0

        # history, filled at each on_epoch_end
        self.epochs: list[int] = []
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self._history: list[np.ndarray] = []  # each entry: per-layer value for that epoch

    def attach(self):
        """No-op: kept for API symmetry. Gradients are read from .grad directly."""
        return self

    def _layer_value(self, params: list[torch.nn.Parameter]) -> float:
        """Combine a layer's parameter gradients into one scalar.

        For L2 we combine sub-tensors correctly: the norm of the concatenation
        equals the sqrt of the sum of squared sub-norms. For mean-abs we take the
        element-count-weighted mean so weight and bias are pooled fairly.
        """
        if self.metric == "l2_norm":
            sq = 0.0
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach()
                    sq += float(torch.sum(g * g).item())
            return float(np.sqrt(sq))
        else:  # mean_abs
            total = 0.0
            count = 0
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach()
                    total += float(torch.sum(torch.abs(g)).item())
                    count += g.numel()
            return total / count if count else 0.0

    def accumulate(self):
        """Call once per optimizer step, after backward(), before zero_grad()."""
        for i, name in enumerate(self.layer_names):
            self._epoch_sum[i] += self._layer_value(self._param_groups[name])
        self._epoch_count += 1

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float | None = None):
        """Finalise the epoch: average the captured values and record losses."""
        if self._epoch_count == 0:
            per_layer = np.zeros(len(self.layer_names), dtype=np.float64)
        else:
            per_layer = self._epoch_sum / self._epoch_count
        self._history.append(per_layer)
        self.epochs.append(epoch)
        self.train_loss.append(float(train_loss))
        self.val_loss.append(float(val_loss) if val_loss is not None else np.nan)
        # reset accumulator
        self._epoch_sum = np.zeros(len(self.layer_names), dtype=np.float64)
        self._epoch_count = 0

    # -- data access ----------------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Return the [num_layers x num_epochs] gradient-magnitude matrix."""
        if not self._history:
            raise RuntimeError("No gradient data captured. Did you call accumulate()/on_epoch_end()?")
        return np.stack(self._history, axis=1)  # rows=layers, cols=epochs

    # -- plots ----------------------------------------------------------------

    def plot_heatmap(self, save_path: str | None = None, log_scale: bool = True):
        """Layers x epochs heatmap with train/val loss overlaid."""
        mat = self.matrix()
        plot_mat = np.log10(mat + 1e-12) if log_scale else mat

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(plot_mat, aspect="auto", origin="lower",
                       extent=[self.epochs[0], self.epochs[-1], -0.5, len(self.layer_names) - 0.5],
                       cmap="viridis")
        ax.set_yticks(range(len(self.layer_names)))
        ax.set_yticklabels(self.layer_names, fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("layer (input -> output)")
        cbar = fig.colorbar(im, ax=ax, pad=0.12)
        cbar.set_label(f"{'log10 ' if log_scale else ''}{self.metric} of gradient")

        # loss overlay on a twin y-axis
        ax2 = ax.twinx()
        ax2.plot(self.epochs, self.train_loss, color="white", lw=1.8, label="train loss")
        if not np.all(np.isnan(self.val_loss)):
            ax2.plot(self.epochs, self.val_loss, color="orange", lw=1.5, ls="--", label="val loss")
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
        # integrate over epochs to get a single contribution score.
        # np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        contrib = _trapz(mat, x=self.epochs, axis=1)
        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(self.layer_names))
        ax.barh(y, contrib, color="steelblue")
        ax.set_yticks(y)
        ax.set_yticklabels(self.layer_names, fontsize=8)
        ax.invert_yaxis()  # input layer on top
        ax.set_xlabel(f"integrated {self.metric} over training")
        ax.set_title("Layer contribution score")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig
