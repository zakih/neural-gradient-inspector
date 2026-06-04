"""Damped harmonic oscillator PINN — a worked example of using the tracker.

Everything here is ordinary PyTorch. Nothing is imported by the library; this is
what *using* the two tools looks like on a real physics-informed ML problem.

Governing ODE:  m x'' + c x' + k x = 0,  x(0)=x0, x'(0)=v0
Underdamped regime (c^2 < 4mk) → decaying sinusoid, which has a closed form we
can check the network against.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Model: a plain tanh MLP. tanh is the PINN default because forming the ODE
# residual differentiates the network output twice via autograd.
# --------------------------------------------------------------------------- #
class OscillatorMLP(nn.Module):
    def __init__(self, hidden_width=64, hidden_depth=4):
        super().__init__()
        layers, prev = [], 1
        for _ in range(hidden_depth):
            layers += [nn.Linear(prev, hidden_width), nn.Tanh()]
            prev = hidden_width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


# --------------------------------------------------------------------------- #
# "Data": collocation points sampled from the time domain. No labels — the
# physics is enforced through the loss.
# --------------------------------------------------------------------------- #
def make_collocation(t_max, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, 1, generator=g) * t_max
    return t


def analytical_solution(t, p):
    """Closed-form x(t) for the underdamped oscillator. Raises if params are not
    underdamped, rather than silently returning a wrong curve."""
    m, c, k, x0, v0 = p["m"], p["c"], p["k"], p["x0"], p["v0"]
    if c * c - 4 * m * k >= 0:
        raise ValueError(f"Underdamped only (c^2<4mk); got c^2={c*c}, 4mk={4*m*k}")
    zeta = c / (2 * math.sqrt(m * k))
    w0 = math.sqrt(k / m)
    wd = w0 * math.sqrt(1 - zeta * zeta)
    decay = -zeta * w0
    A = x0
    B = (v0 - decay * x0) / wd
    return np.exp(decay * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))


# --------------------------------------------------------------------------- #
# Physics loss: PDE residual + initial conditions. Their differing magnitudes
# are the classic PINN gradient pathology — exactly what the heatmap reveals.
# --------------------------------------------------------------------------- #
def _d(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


def physics_loss(model, t, p, w_physics=1.0, w_ic=1.0):
    t = t.clone().requires_grad_(True)
    x = model(t)
    x_t = _d(x, t)
    x_tt = _d(x_t, t)
    residual = p["m"] * x_tt + p["c"] * x_t + p["k"] * x
    loss_pde = torch.mean(residual ** 2)

    t0 = torch.zeros(1, 1, requires_grad=True)
    x0 = model(t0)
    v0 = _d(x0, t0)
    loss_ic = (x0 - p["x0"]) ** 2 + (v0 - p["v0"]) ** 2
    loss_ic = torch.mean(loss_ic)

    total = w_physics * loss_pde + w_ic * loss_ic
    return total, loss_pde.detach(), loss_ic.detach()
