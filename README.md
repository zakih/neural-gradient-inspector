# PINN Gradient-Flow Framework

A modular PyTorch training framework for **physics-informed neural networks (PINNs)**, built around a gradient-flow inspection tool that shows *where and when learning actually happens inside the network during training*.

The included example trains a PINN to solve the **1D damped harmonic oscillator** — no dataset download, trains on a laptop CPU in under a minute, and reproduces the closed-form analytical solution. Swap in your own `BaseModel` and `BaseDataset` to use the framework for any problem.

---

## What This Is

This repository is built around three principles:

**Modularity** — model, dataset, and training configuration are fully decoupled. Swap any component without touching the rest of the infrastructure.

**Observability** — every run is logged, versioned, and reproducible. MLflow tracks all metrics, hyperparameters, and artifacts automatically.

**Insight** — the gradient-flow visualizer reveals the per-layer learning dynamics across epochs, with training/validation loss overlaid. For PINNs in particular this exposes the well-known tension between the PDE-residual and initial-condition loss terms.

---

## The Gradient-Flow Inspection Component

The core question this tool answers: **as training proceeds, where in the network is learning actually happening — and does it stay healthy?**

Standard tooling tells you *that* a model converged. It rarely tells you *which layers drove that convergence*, whether early layers stalled, or whether the gradient signal degraded propagating back through the network. This component makes that visible.

**How it works.** `GradientTracker` reads each layer's parameter gradients after every backward pass and records a single scalar summary per layer — the **L2 norm** of that layer's gradient (mean-absolute is also available). Recording one statistic per layer rather than every gradient element keeps memory flat (a handful of floats per step) and makes the tool usable on large models where storing full gradient tensors would be infeasible. Captures are aggregated per epoch.

> **Why the L2 norm and not a signed sum:** positive and negative gradient elements cancel in a sum, so a hard-learning layer can read as ≈0. The norm measures the magnitude of the update signal, which is what "is this layer learning" actually means.

The result is a compact `[num_layers × num_epochs]` matrix that drives three views:

**1. Per-epoch gradient heatmap** — epochs on the x-axis, layers (input → output) on the y-axis, color encoding gradient magnitude (log scale by default). Training and validation loss are overlaid on a twin axis so gradient behavior reads directly against convergence. Surfaces at a glance:
- **Vanishing gradients** — layers that go dark early and stop learning
- **Exploding gradients** — layers that spike into instability
- **Learning dynamics** — which layers learn fast, which lag, and how that shifts
- **Architectural effects** — the smoothing impact of normalization or residual connections

**2. Gradient norm curves** — per-layer gradient-norm trajectories over epochs, one line per layer: did a layer stabilize, collapse, or oscillate?

**3. Layer contribution score** — a bar chart of each layer's gradient magnitude integrated over the whole run, ranking which layers drove the most learning.

All three plots save to the active MLflow run's artifact directory automatically.

```python
from visualizations import GradientTracker

tracker = GradientTracker(model, metric="l2_norm")  # or "mean_abs"

for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(...)
        loss.backward()
        tracker.accumulate()        # read .grad, after backward, before zero_grad
        optimizer.step()
    tracker.on_epoch_end(epoch, train_loss, val_loss)   # finalise epoch + losses

tracker.plot_heatmap("runs/gradient_heatmap.png")
tracker.plot_curves("runs/gradient_curves.png")
tracker.plot_contributions("runs/layer_contributions.png")
```

*Roadmap:* an animated epoch-by-epoch view, and correlation of gradient magnitude with input-data characteristics (which inputs drive learning in which layers).

---

## Why a PINN Example

A physics-informed network is an ideal demonstration of the gradient tool. Its loss is a sum of competing terms — a PDE-residual term enforced at collocation points and an initial/boundary-condition term — and these terms routinely produce gradients of very different magnitudes. That imbalance is a documented PINN training pathology, and it is exactly the kind of structure the gradient heatmap is built to reveal. The damped oscillator also has a closed-form solution, so the network's output can be checked against ground truth.

The damped harmonic oscillator:

$$m\,x'' + c\,x' + k\,x = 0, \qquad x(0)=x_0,\ x'(0)=v_0$$

In the underdamped regime ($c^2 < 4mk$) the solution is a decaying sinusoid. The network is a small tanh MLP; the ODE residual is formed by differentiating the network output twice with `torch.autograd.grad`.

---

## Repository Structure

```
pinn-gradient-flow/
├── README.md
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── train.py                    # CLI entry point
│
├── notebooks/
│   └── training.ipynb          # Main walkthrough — start here
│
├── src/
│   ├── config.py               # Typed dataclass config + YAML + CLI overrides
│   ├── dataset.py              # BaseDataset + OscillatorDataset (+ analytical solution)
│   ├── model.py                # BaseModel + configurable MLP
│   ├── trainer.py              # Training loop, PINN loss, MLflow integration
│   └── utils.py                # Seeding, device detection, checkpointing
│
├── visualizations/
│   ├── gradients.py            # GradientTracker — the inspection component
│   └── architecture.py         # Model → Mermaid diagram generator
│
├── configs/
│   └── default.yaml            # Default configuration
│
├── data/ runs/ checkpoints/    # Mounted / generated artifacts
```

---

## Quickstart

### Docker (recommended)

```bash
git clone <your-repo-url>
cd pinn-gradient-flow
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up
```

- JupyterLab → http://localhost:8888 (open `notebooks/training.ipynb`)
- MLflow UI → http://localhost:5000

Passing `USER_ID`/`GROUP_ID` makes files written into the mounted volume owned by your host user rather than root.

### Local

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python train.py
```

`train.py` accepts dotted CLI overrides:

```bash
python train.py training.epochs=500 physics.c=0.1 visualization.gradient_metric=mean_abs
```

---

## Plugging In Your Own Model and Dataset

**1.** Subclass `BaseDataset` (`src/dataset.py`) — implement `__len__` and `__getitem__`.
**2.** Subclass `BaseModel` (`src/model.py`) — implement `forward`.
**3.** Adjust `configs/default.yaml`.
**4.** Run `train.py` or the notebook.

The training loop, logging, checkpointing, and gradient visualization run unchanged. (For a non-PINN problem, replace the physics loss in `src/trainer.py` with a standard supervised loss — the gradient tracker is loss-agnostic.)

---

## MLflow Tracking

Every run logs all hyperparameters, per-epoch losses (total / physics / IC), learning rate, the config snapshot, the architecture diagram, all gradient plots, the solution-vs-analytical comparison, the best metric and epoch, training duration, and a checkpoint.

```bash
mlflow ui --backend-store-uri runs/mlflow   # then open http://localhost:5000
```

---

## Configuration Reference

See `configs/default.yaml`. Key sections: `physics` (oscillator parameters), `model` (MLP width/depth/activation), `training` (epochs, LR, scheduler, loss weights `w_physics`/`w_ic`, gradient clipping), `visualization` (capture interval, gradient metric), and `mlflow`.

---

## Requirements

PyTorch (CPU build for this demo), MLflow, matplotlib, seaborn, scipy, numpy, pyyaml, tqdm, jupyterlab. See `requirements.txt` (torch is installed separately from the CPU wheel index — see the file's header note).

---

## Contributing

Issues and PRs welcome, particularly: additional schedulers, alternative logging backends (wandb, tensorboard), distributed training (DDP), the animated gradient view, and additional PDE examples (heat equation, Burgers').
