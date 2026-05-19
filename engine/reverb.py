"""
engine/reverb.py
----------------
Convolution reverb for ocean spaces (beach, cove, underwater).
Falls back to algorithmic reverb if no IR files available.
"""

import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve

from . import config

_IR_DIR = Path(__file__).parent / "impulse_responses"
_ir_cache: dict = {}

AVAILABLE_SPACES = ["beach", "cove", "underwater", "cliff", "cave"]


def get_available_spaces() -> list:
    """Return list of available IR space names."""
    found = [f.stem for f in _IR_DIR.glob("*.wav")]
    return found if found else AVAILABLE_SPACES


def _generate_synthetic_ir(space: str, duration: float = 3.0) -> np.ndarray:
    """Generate a synthetic impulse response when no WAV file is available."""
    n_samples = int(duration * config.SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Different decay characteristics per space
    params = {
        "beach":      {"decay": 1.5, "diffusion": 0.8, "color": "warm"},
        "cove":       {"decay": 2.5, "diffusion": 0.6, "color": "dark"},
        "underwater": {"decay": 4.0, "diffusion": 0.9, "color": "very_dark"},
        "cliff":      {"decay": 2.0, "diffusion": 0.4, "color": "bright"},
        "cave":       {"decay": 3.5, "diffusion": 0.7, "color": "dark"},
    }
    p = params.get(space, params["beach"])

    # Exponential decay with noise
    decay_env = np.exp(-t * (3.0 / p["decay"]))
    noise = np.random.randn(n_samples)

    # Apply color (frequency shaping)
    from .filters import Filters
    if p["color"] == "very_dark":
        noise = Filters.lowpass(noise, 800, order=3)
    elif p["color"] == "dark":
        noise = Filters.lowpass(noise, 2000, order=2)
    elif p["color"] == "warm":
        noise = Filters.lowpass(noise, 4000, order=2)

    # Add some early reflections
    ir = noise * decay_env * p["diffusion"]

    # Early reflection spikes
    n_reflections = int(4 + 6 * p["diffusion"])
    for i in range(n_reflections):
        pos = int(np.random.uniform(0.01, 0.15) * config.SAMPLE_RATE)
        if pos < n_samples:
            ir[pos] += np.random.uniform(0.3, 0.8) * (0.8 ** i)

    # Normalize
    ir /= np.max(np.abs(ir)) + 1e-10
    return ir


def load_ir(space: str) -> np.ndarray:
    """Load IR from file or generate synthetic one."""
    if space in _ir_cache:
        return _ir_cache[space]

    ir_path = _IR_DIR / f"{space}.wav"
    if ir_path.exists():
        try:
            import soundfile as sf
            ir_data, ir_sr = sf.read(str(ir_path), dtype="float32")
            if ir_sr != config.SAMPLE_RATE:
                from scipy.signal import resample
                n_target = int(len(ir_data) * config.SAMPLE_RATE / ir_sr)
                ir_data = resample(ir_data, n_target)
            if ir_data.ndim > 1:
                ir_data = ir_data.mean(axis=1)
            _ir_cache[space] = ir_data.astype(np.float64)
        except Exception:
            _ir_cache[space] = _generate_synthetic_ir(space)
    else:
        _ir_cache[space] = _generate_synthetic_ir(space)

    return _ir_cache[space]


def apply_reverb(
    audio: np.ndarray,
    space: str = "beach",
    mix: float = 0.3,
    decay_trim: float = 1.0,
) -> np.ndarray:
    """
    Apply convolution reverb to audio.
    audio: mono (N,) or stereo (N,2)
    """
    ir = load_ir(space)

    # Trim IR for shorter/longer tails
    if decay_trim != 1.0:
        trim_len = int(len(ir) * decay_trim)
        ir = ir[:trim_len]

    if audio.ndim == 2:
        # Process each channel
        wet_l = fftconvolve(audio[:, 0], ir, mode='full')[:len(audio)]
        wet_r = fftconvolve(audio[:, 1], ir, mode='full')[:len(audio)]
        wet = np.column_stack([wet_l, wet_r])
    else:
        wet = fftconvolve(audio, ir, mode='full')[:len(audio)]

    # Normalize wet signal
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet = wet / peak * np.max(np.abs(audio))

    return audio * (1.0 - mix) + wet * mix
