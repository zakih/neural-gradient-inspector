"""Typed configuration for the PINN training framework.

Configs are dataclasses (no magic strings, IDE autocomplete, type checks) that
load from a YAML file and accept dotted-key CLI overrides. The fully-resolved
config is snapshotted into every MLflow run for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Any
import yaml


def _parse_scalar(value: str) -> Any:
    """Parse a CLI override value to its natural type.

    YAML 1.1 (what PyYAML's safe_load implements) does NOT parse unsigned
    scientific notation like '5e-4' as a float -- it requires a decimal point,
    e.g. '5.0e-4'. Since learning rates are routinely written '1e-3', we try a
    plain float conversion first and fall back to YAML for everything else
    (ints, bools, strings, lists).
    """
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return yaml.safe_load(value)


@dataclass
class ExperimentConfig:
    name: str = "damped_oscillator_pinn"
    seed: int = 42


@dataclass
class PhysicsConfig:
    """Parameters of the 1D damped harmonic oscillator.

    Governing ODE:   m x'' + c x' + k x = 0
    Initial cond.:   x(0) = x0,  x'(0) = v0
    Domain:          t in [0, t_max]

    The defaults sit in the underdamped regime (c^2 < 4 m k), which gives the
    decaying-sinusoid solution that is visually unmistakable against the
    network's prediction.
    """
    m: float = 1.0          # mass
    c: float = 0.3          # damping coefficient
    k: float = 4.0          # stiffness
    x0: float = 1.0         # initial position
    v0: float = 0.0         # initial velocity
    t_max: float = 20.0     # length of the time domain
    n_collocation: int = 2000   # interior points where the PDE residual is enforced


@dataclass
class ModelConfig:
    in_dim: int = 1
    out_dim: int = 1
    hidden_width: int = 64
    hidden_depth: int = 4        # number of hidden layers
    activation: str = "tanh"     # tanh is standard for PINNs (smooth, infinitely differentiable)


@dataclass
class TrainingConfig:
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "cosine"            # one of: none, step, cosine, plateau
    early_stopping_patience: int = 0     # 0 disables early stopping
    gradient_clip_norm: float = 0.0      # 0 disables clipping
    # Relative weighting of the two competing loss terms. The imbalance between
    # these is the gradient pathology the visualizer is designed to expose.
    w_physics: float = 1.0
    w_ic: float = 1.0


@dataclass
class CheckpointConfig:
    save_best: bool = True
    save_every_n_epochs: int = 50
    metric: str = "val_loss"
    mode: str = "min"


@dataclass
class VisualizationConfig:
    gradient_capture_interval: int = 1   # capture every N epochs
    gradient_metric: str = "l2_norm"     # l2_norm or mean_abs
    architecture_diagram: bool = True
    save_all_plots: bool = True


@dataclass
class MLflowConfig:
    tracking_uri: str = "runs/mlflow"
    experiment_name: str = "default"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # ---- construction helpers -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            section_cls = f.type if is_dataclass(f.type) else None
            # f.type may arrive as a string under `from __future__ import annotations`;
            # resolve against this module's globals.
            if isinstance(section_cls, str) or section_cls is None:
                section_cls = globals()[f.default_factory().__class__.__name__]
            section_raw = raw.get(f.name, {}) or {}
            kwargs[f.name] = section_cls(**section_raw)
        return cls(**kwargs)

    def apply_overrides(self, overrides: list[str]) -> "Config":
        """Apply CLI-style dotted overrides, e.g. ['training.epochs=500'].

        Values are parsed as YAML scalars so 1e-3, true, 42, and bare strings
        all land with the right type.
        """
        d = self.to_dict()
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override '{item}' must be of the form section.key=value")
            key, _, value = item.partition("=")
            parts = key.strip().split(".")
            parsed = _parse_scalar(value.strip())
            cursor = d
            for p in parts[:-1]:
                if p not in cursor:
                    raise KeyError(f"Unknown config section '{p}' in override '{item}'")
                cursor = cursor[p]
            leaf = parts[-1]
            if leaf not in cursor:
                raise KeyError(f"Unknown config key '{leaf}' in override '{item}'")
            cursor[leaf] = parsed
        return Config.from_dict(d)

    # ---- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten to 'section.key' -> value for MLflow param logging."""
        flat: dict[str, Any] = {}
        for section, params in self.to_dict().items():
            for k, v in params.items():
                flat[f"{section}.{k}"] = v
        return flat

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
