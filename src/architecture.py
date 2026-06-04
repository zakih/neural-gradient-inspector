"""Generate a Mermaid diagram of any PyTorch model by runtime inspection.

Standalone utility -- give it any nn.Module, get back a Mermaid `graph TD` string
that renders natively in GitHub Markdown and Jupyter. No assumptions about how
your model is defined.

    from src.architecture import to_mermaid
    print(to_mermaid(model, input_shape=(1, 784)))
"""

from __future__ import annotations

import torch.nn as nn


def _describe(module: nn.Module) -> str | None:
    """One-line label for a leaf module, or None to skip it."""
    if isinstance(module, nn.Linear):
        return f"Linear\\n{module.in_features} → {module.out_features}"
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return f"{module.__class__.__name__}\\n{module.in_channels} → {module.out_channels}"
    if isinstance(module, nn.Dropout):
        return f"Dropout\\np={module.p}"
    cls = module.__class__.__name__
    if isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU, nn.Sigmoid, nn.LeakyReLU)):
        return cls
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        return cls
    # custom leaf with no children (e.g. a Sine activation): name it
    if len(list(module.children())) == 0:
        return cls
    return None


def to_mermaid(model: nn.Module, input_shape=None) -> str:
    """Return a Mermaid `graph TD` string for the model.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple, optional
        If given, adds Input/Output nodes labelled with the shape.
    """
    nodes: list[tuple[str, str]] = []
    idx = 0

    if input_shape is not None:
        shape_str = "[" + ", ".join(str(s) for s in input_shape) + "]"
        nodes.append((f"N{idx}", f"Input\\n{shape_str}"))
        idx += 1

    for _, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # container, not a leaf
        label = _describe(module)
        if label is None:
            continue
        nodes.append((f"N{idx}", label))
        idx += 1

    if input_shape is not None:
        nodes.append((f"N{idx}", "Output"))

    lines = ["graph TD"]
    for nid, label in nodes:
        lines.append(f'    {nid}["{label}"]')
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        lines.append(f"    {a} --> {b}")
    return "\n".join(lines)


def save_mermaid(model: nn.Module, path: str, input_shape=None) -> str:
    """Write the Mermaid source to a .mmd file and return it."""
    src = to_mermaid(model, input_shape)
    with open(path, "w") as f:
        f.write(src)
    return src


def save_architecture_png(model, path, input_shape=None, title="Network architecture"):
    """Render the model's layer stack as a labeled boxes-and-arrows PNG.

    Dependency-free (matplotlib only): no Mermaid CLI, no browser, no network.
    Walks the same leaf modules as `to_mermaid`, color-coded by layer type.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import torch.nn as nn

    nodes = []
    if input_shape is not None:
        nodes.append(("Input  [" + ", ".join(str(s) for s in input_shape) + "]", "io"))

    for _, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # container, not a leaf
        cls = module.__class__.__name__
        if isinstance(module, nn.Linear):
            nodes.append((f"Linear   {module.in_features} → {module.out_features}", "linear"))
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nodes.append((f"{cls}   {module.in_channels} → {module.out_channels}", "linear"))
        elif isinstance(module, nn.Dropout):
            nodes.append((f"Dropout  p={module.p}", "other"))
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU, nn.Sigmoid, nn.LeakyReLU)):
            nodes.append((cls, "act"))
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            nodes.append((cls, "norm"))
        elif len(list(module.children())) == 0:
            nodes.append((cls, "act"))  # custom activation-like leaf
    if input_shape is not None:
        nodes.append(("Output", "io"))

    colors = {"io": "#d9d9d9", "linear": "#4e79a7", "act": "#59a14f",
              "norm": "#edc948", "other": "#b07aa1"}
    text_colors = {"io": "#222", "linear": "#fff", "act": "#fff",
                   "norm": "#222", "other": "#fff"}

    n = len(nodes)
    box_h, gap = 0.62, 0.38
    fig_h = max(2.5, n * (box_h + gap))
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    ax.set_xlim(0, 4); ax.set_ylim(0, n * (box_h + gap) + 0.2)
    ax.axis("off")

    y = n * (box_h + gap) - box_h
    centers = []
    for label, kind in nodes:
        box = FancyBboxPatch((0.8, y), 2.4, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=0, facecolor=colors[kind])
        ax.add_patch(box)
        ax.text(2.0, y + box_h / 2, label, ha="center", va="center",
                fontsize=9, color=text_colors[kind], weight="medium")
        centers.append(y + box_h / 2)
        y -= (box_h + gap)

    for top, bot in zip(centers, centers[1:]):
        ax.annotate("", xy=(2.0, bot + box_h / 2 + 0.02),
                    xytext=(2.0, top - box_h / 2 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))

    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig