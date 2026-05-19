"""
engine/preset_loader.py
-----------------------
Load and validate YAML ocean presets.
"""

import copy
from pathlib import Path

import yaml

from . import config


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Default layer templates
_WAVE_DEFAULTS = {
    "type": "wave",
    "enabled": True,
    "mix": 1.0,
    "pan": -0.2,
    "noise": {"white": 0.05, "pink": 0.35, "brown": 0.6},
    "frequency": {"low": 60, "high": 1200, "order": 3},
    "envelope": {"cycle_period": 8.0, "attack_ratio": 0.45, "randomize": 0.3},
}

_RISACCA_DEFAULTS = {
    "type": "risacca",
    "enabled": True,
    "mix": 0.7,
    "pan": 0.2,
    "noise": {"white": 0.2, "pink": 0.5, "brown": 0.3},
    "frequency": {"low": 500, "high": 6000, "order": 3},
    "envelope": {"cycle_period": 6.0, "delay": 2.0},
}

_FOAM_DEFAULTS = {
    "type": "foam",
    "enabled": True,
    "mix": 0.3,
    "pan": 0.0,
    "frequency": {"low": 3000, "high": 12000},
    "envelope": {"cycle_period": 7.0},
}

_LAYER_DEFAULTS = {
    "wave": _WAVE_DEFAULTS,
    "risacca": _RISACCA_DEFAULTS,
    "foam": _FOAM_DEFAULTS,
}


def load_preset(path: Path) -> dict:
    """Load a YAML preset and normalize it for the engine."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Empty preset: {path}")

    # Normalize duration
    if "duration" not in raw:
        global_cfg = raw.get("global", {})
        raw["duration"] = float(global_cfg.get("duration_seconds", 60))

    # Normalize layers — merge with defaults
    layers = raw.get("layers", [])
    normalized_layers = []
    for layer in layers:
        layer_type = layer.get("type", "wave")
        defaults = _LAYER_DEFAULTS.get(layer_type, {})
        merged = _deep_merge(defaults, layer)
        normalized_layers.append(merged)
    raw["layers"] = normalized_layers

    # Ensure meta exists
    if "meta" not in raw:
        raw["meta"] = {"name": path.stem, "slug": path.stem.lower().replace(" ", "_")}

    return raw
