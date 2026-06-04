"""Models for the PINN framework.

The example model is a plain MLP -- for a PINN the architecture is deliberately
simple; the physics lives in the loss, not the network. tanh activations are the
PINN default because forming the ODE residual requires differentiating the
network output twice w.r.t. its input, and tanh is smooth to all orders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Subclass this and implement forward()."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "sin": None,  # handled specially below
}


class _Sine(nn.Module):
    """Sine activation (SIREN-style); occasionally useful for oscillatory PINNs."""
    def forward(self, x):
        return torch.sin(x)


class MLP(BaseModel):
    """Fully-connected network: in_dim -> [hidden_width] * hidden_depth -> out_dim."""

    def __init__(self, config):
        super().__init__()
        act_name = config.activation.lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{config.activation}'. "
                             f"Choose from {list(_ACTIVATIONS)}.")

        def make_act():
            return _Sine() if act_name == "sin" else _ACTIVATIONS[act_name]()

        layers: list[nn.Module] = []
        prev = config.in_dim
        for _ in range(config.hidden_depth):
            layers.append(nn.Linear(prev, config.hidden_width))
            layers.append(make_act())
            prev = config.hidden_width
        layers.append(nn.Linear(prev, config.out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # Xavier init suits tanh/sigmoid-family activations and keeps the early
        # forward/backward signal well-scaled -- relevant since the gradient
        # visualizer is reading exactly that signal.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
