"""Generate a Mermaid diagram of a model's architecture by runtime inspection.

Walks the model's leaf modules in definition order and emits a top-down Mermaid
`graph TD`. Linear layers show their in/out dimensions; activations and dropout
are labelled by type. The string renders natively in GitHub Markdown and Jupyter.
"""

from __future__ import annotations

import torch.nn as nn


def _describe(module: nn.Module) -> str | None:
    """One-line label for a leaf module, or None to skip it."""
    if isinstance(module, nn.Linear):
        return f"Linear\\n{module.in_features} → {module.out_features}"
    if isinstance(module, nn.Dropout):
        return f"Dropout\\np={module.p}"
    cls = module.__class__.__name__
    # activations and norms: just name them
    if isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU, nn.Sigmoid)):
        return cls
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        return cls
    # custom leaves (e.g. the _Sine activation)
    if len(list(module.children())) == 0 and any(True for _ in module.parameters(recurse=False)) is False:
        return cls
    return None


def build_mermaid(model: nn.Module, input_shape=None) -> str:
    """Return a Mermaid `graph TD` string for the model."""
    nodes: list[tuple[str, str]] = []  # (node_id, label)
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


def plot_architecture(model: nn.Module, input_shape=None, save_path: str | None = None) -> str:
    """Build the Mermaid diagram, optionally save it, and return the source string.

    In Jupyter the returned string can be rendered with the `%%mermaid` magic or
    by writing it into a Markdown cell fenced as ```mermaid.
    """
    mermaid = build_mermaid(model, input_shape)
    if save_path:
        with open(save_path, "w") as f:
            f.write(mermaid)
    return mermaid
