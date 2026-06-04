"""Reproducibility, device management, and checkpointing helpers."""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN; on CPU-only runs these are simply no-ops.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect CUDA, else CPU. (Apple MPS is intentionally not used here:
    second-order autograd for the PINN residual is more reliable on CPU/CUDA.)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
    return "cpu"


def save_checkpoint(path: str, model, optimizer, epoch: int, best_metric: float,
                    config_dict: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
