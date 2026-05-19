"""
engine/noise.py
---------------
Noise generators: white, pink, brown (ocean-realistic spectra).
"""

import numpy as np
from . import config


def white_noise(n_samples: int) -> np.ndarray:
    return np.random.randn(n_samples)


def pink_noise(n_samples: int) -> np.ndarray:
    """Generate pink noise (1/f) using the Voss-McCartney algorithm."""
    # Use 16 octaves for smooth 1/f slope
    n_octaves = 16
    # Pad to power of 2 for efficiency
    n = max(n_samples, 1024)

    white = np.random.randn(n_octaves, n)
    # Weight by 1/sqrt(octave) for 1/f spectrum
    weights = 1.0 / np.sqrt(np.arange(1, n_octaves + 1))

    result = np.zeros(n)
    for i in range(n_octaves):
        # Each octave has samples held for 2^i steps
        step = 2 ** i
        held = np.repeat(white[i, :n // step + 1], step)[:n]
        result += held * weights[i]

    result = result[:n_samples]
    result /= np.max(np.abs(result)) + 1e-10
    return result


def brown_noise(n_samples: int) -> np.ndarray:
    """Generate brown noise (1/f²) via integrated white noise."""
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    # Remove DC drift with highpass-like normalization
    brown -= np.linspace(brown[0], brown[-1], n_samples)
    brown /= np.max(np.abs(brown)) + 1e-10
    return brown


def noise_blend(n_samples: int, white_mix: float = 0.1, pink_mix: float = 0.5, brown_mix: float = 0.4) -> np.ndarray:
    """Blend noise types for ocean-like spectrum."""
    total = white_mix + pink_mix + brown_mix
    if total == 0:
        return np.zeros(n_samples)
    w = white_noise(n_samples) * white_mix / total
    p = pink_noise(n_samples) * pink_mix / total
    b = brown_noise(n_samples) * brown_mix / total
    return w + p + b
