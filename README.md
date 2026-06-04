# Model Gradient Tracker

If you are used to designing neural network architectures layer by layer, and feel unsatisfied by just looking at the loss curves, then this tool can help you visualize the magnitude of the gradient over the training epochs and the various layers. This closer inspection of the graident can help diagnose various training scenarios. Two tools in this repo:

1. **`GradientTracker`** — records per-layer gradient magnitude across epochs and plots it. It peels back one layer beyond the loss curve: instead of only knowing *that* training converged, we see *where* in the network, across depth and across epochs, the learning actually occurs.
2. Model architecture visualizer using **`to_mermaid` and/or `save_mermaid`** which turns any `nn.Module` into a Mermaid diagram (you'll have to render it yourself if you want a PNG of it), and `save_architecture_png` which is an equivalent matplotlib version.

The library imposes nothing on how the model, data, or training loop are written. The tools can be called inside typical training loops (see below and `/examples`).

The `examples/oscillator/` directory shows both tools applied to a physics-informed neural network solving the 1D damped harmonic oscillator.

## Future work
I want to add some metadata hooks to see which data batches (if any) stand out in their contribution to the gradient. 


## The Tools

### GradientTracker

```python
from src.gradient_tracker import GradientTracker

tracker = GradientTracker(model, metric="l2_norm")   # any nn.Module

for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        loss = your_loss(...)
        loss.backward()
        tracker.accumulate()                          # after backward, before zero_grad
        optimizer.step()
    tracker.log_epoch(epoch, losses={"train": train_loss, "val": val_loss})

tracker.plot_heatmap("heatmap.png")
tracker.plot_curves("curves.png")
tracker.plot_contributions("contributions.png")
```

It records one scalar per layer per step — the gradient **L2 norm** (or `mean_abs`) — building a compact `[layers × epochs]` matrix. One statistic per layer rather than every gradient element keeps memory flat and works on large models.

> *Why the norm, not a signed sum:* positive and negative gradient elements cancel in a sum, so a hard-learning layer can read as ≈0. The norm measures the magnitude of the update signal — what "is this layer learning" actually means.

Three views:
- **Heatmap** — layers (input→output) × epochs, color = gradient magnitude, with any losses you logged overlaid. Reveals vanishing/exploding gradients, which layers learn fast or stall, and the effect of architectural choices.
- **Curves** — per-layer gradient-norm trajectories.
- **Contributions** — each layer's gradient magnitude integrated over the run, ranked.

### to_mermaid

```python
from src.architecture import to_mermaid
print(to_mermaid(model, input_shape=(1, 784)))
```

Returns a Mermaid `graph TD` string. Paste it into a Markdown cell fenced as ` ```mermaid ` to render it.



## Repository Structure

```
gradient-tracker/
├── README.md
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
│
├── src/              
│   ├── __init__.py
│   ├── gradient_tracker.py
│   └── architecture.py
│
├── templates/
│   └── training_template.ipynb  ← copy this to start a new project (It's a fill-in-the-blank style file)
│
└── examples/
    └── oscillator/              ← worked example: PINN for the damped oscillator
        ├── oscillator.py
        ├── config.yaml
        └── oscillator.ipynb
```

`src/` is the whole library. Everything in `templates/` is to help getting started with using the tools.



## Quickstart



```bash
git clone <your-repo-url>
cd gradient-tracker
```

### Docker

- build the container
``` shell
docker compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
```

- run the container
``` shell
docker compose up -d
```

- stop the container
``` shell
docker compose down
```

Open **http://localhost:8888** for JupyterLab, then run `examples/oscillator/oscillator.ipynb`. (Passing `USER_ID`/`GROUP_ID` keeps generated files owned by you, not root.). Or attach to container in VSCode.

### Local

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter lab
```


## Using It in Your Own Project

Copy `templates/training_template.ipynb`. Fill in three TODOs — your model, your data, your loss. The tracker calls and the generic loop are already wired in. You import from `src/` and never touch it.


## Example: 1D Oscillator Physics-Informed Neural Network (PINN)

A physics-informed neural network is an ideal showcase: its loss combines a PDE-residual term and an initial-condition term, and these routinely produce gradients of very different magnitudes — a documented PINN training pathology that the gradient heatmap makes visible. The damped oscillator also has a closed-form solution, so the network's output is checked against ground truth.

$$m\,x'' + c\,x' + k\,x = 0,\qquad x(0)=x_0,\ x'(0)=v_0$$

No dataset download — the "data" is collocation points sampled from the time domain; the physics lives in the loss. Trains on a laptop CPU in under a minute. See `examples/oscillator/`.


## Requirements

See `requirements.txt`. Note, Pytorch CPU is installed just for the oscillator demo.

