"""Datasets for the PINN framework.

For a physics-informed network the "data" is not labelled examples -- it is a
cloud of collocation points sampled from the problem domain at which the
governing equation's residual is enforced. This module provides:

  * BaseDataset      -- the interface users subclass for their own problem
  * OscillatorDataset -- collocation points for the 1D damped oscillator, plus
                         the closed-form analytical solution for evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class BaseDataset(Dataset, ABC):
    """Subclass this and implement __len__ and __getitem__."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx):
        ...


class OscillatorDataset(BaseDataset):
    """Collocation points t in [0, t_max] for the damped harmonic oscillator.

    Each item is a single time coordinate (shape [1]). The PDE residual loss is
    evaluated at these points; the physics needs no labels, so __getitem__
    returns just the input coordinate. Ground-truth positions are available via
    `analytical_solution` for plotting and test-set error.
    """

    def __init__(self, physics):
        self.p = physics
        # Sample collocation points. Endpoint t=0 is included so the dataloader
        # domain covers the initial condition region; the IC itself is enforced
        # separately as a hard term in the loss.
        t = np.linspace(0.0, physics.t_max, physics.n_collocation, dtype=np.float32)
        self.t = torch.from_numpy(t).unsqueeze(1)  # [N, 1]

    def __len__(self) -> int:
        return self.t.shape[0]

    def __getitem__(self, idx):
        return self.t[idx]

    # -- ground truth ----------------------------------------------------------

    def analytical_solution(self, t: np.ndarray) -> np.ndarray:
        """Closed-form x(t) for m x'' + c x' + k x = 0.

        Implemented for the underdamped regime (c^2 < 4mk), which is the default
        and the only regime that produces the decaying oscillation. Raises if the
        configured parameters fall outside it, rather than silently returning a
        wrong curve.
        """
        m, c, k = self.p.m, self.p.c, self.p.k
        x0, v0 = self.p.x0, self.p.v0
        disc = c * c - 4.0 * m * k
        if disc >= 0:
            raise ValueError(
                f"analytical_solution implemented for the underdamped case only "
                f"(c^2 < 4mk); got c^2={c*c:.4f}, 4mk={4*m*k:.4f}. "
                f"Reduce physics.c or adjust m/k."
            )
        zeta = c / (2.0 * math.sqrt(m * k))          # damping ratio
        w0 = math.sqrt(k / m)                         # natural frequency
        wd = w0 * math.sqrt(1.0 - zeta * zeta)        # damped frequency
        decay = -zeta * w0
        # x(t) = e^{decay t} (A cos wd t + B sin wd t), fit to x(0)=x0, x'(0)=v0
        A = x0
        B = (v0 - decay * x0) / wd
        return np.exp(decay * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))


def make_dataloaders(dataset: BaseDataset, batch_size: int, val_split: float,
                     test_split: float, num_workers: int, seed: int):
    """Split a dataset into train/val/test loaders with a fixed seed.

    Returns (train_loader, val_loader, test_loader). val or test loaders may be
    None if their split is 0.
    """
    n = len(dataset)
    n_val = int(round(n * val_split))
    n_test = int(round(n * test_split))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("val_split + test_split leave no training data")

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    def _loader(ds, shuffle):
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=False)

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
