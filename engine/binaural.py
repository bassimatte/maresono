"""
engine/binaural.py
------------------
Binaural theta beat generator for meditation/relaxation.

Theta frequency range (4-8 Hz) promotes deep relaxation and meditation.
Two modes:
  - "embedded": subtle theta-frequency amplitude modulation on the ocean sound
  - "carrier": dedicated sine pair (L/R slightly detuned) mixed underneath
"""

import numpy as np
from . import config


def generate_theta_modulation(
    n_samples: int,
    beat_hz: float = 6.0,
    depth: float = 0.3,
) -> np.ndarray:
    """
    Generate amplitude modulation envelope at theta frequency.
    Applied to the ocean mix for subtle pulsing.
    """
    t = np.linspace(0, n_samples / config.SAMPLE_RATE, n_samples, endpoint=False)
    modulation = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * beat_hz * t))
    return modulation


def generate_binaural_carrier(
    duration: float,
    carrier_hz: float = 150.0,
    beat_hz: float = 6.0,
    amplitude: float = 0.03,
    fade_secs: float = 8.0,
) -> np.ndarray:
    """
    Generate a stereo binaural carrier tone, shaped to sit underneath
    the ocean sound without being consciously audible.
    Left ear: carrier_hz - beat_hz/2
    Right ear: carrier_hz + beat_hz/2
    Returns (N, 2) stereo array.
    """
    n_samples = int(duration * config.SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    freq_l = carrier_hz - beat_hz / 2.0
    freq_r = carrier_hz + beat_hz / 2.0

    # Use a softer waveform: sine with slight 2nd harmonic for warmth
    left = (np.sin(2 * np.pi * freq_l * t) * 0.85
            + np.sin(2 * np.pi * freq_l * 2 * t) * 0.15) * amplitude
    right = (np.sin(2 * np.pi * freq_r * t) * 0.85
             + np.sin(2 * np.pi * freq_r * 2 * t) * 0.15) * amplitude

    # Slow amplitude breathing so it doesn't feel static
    breath_rate = 0.07  # very slow ~14s cycle
    breath = 0.7 + 0.3 * np.sin(2 * np.pi * breath_rate * t)
    left *= breath
    right *= breath

    # Long fade in/out to avoid any noticeable onset
    fade_samples = int(fade_secs * config.SAMPLE_RATE)
    fade_samples = min(fade_samples, n_samples // 3)
    fade_in = np.linspace(0, 1, fade_samples) ** 3
    fade_out = np.linspace(1, 0, fade_samples) ** 3
    left[:fade_samples] *= fade_in
    left[-fade_samples:] *= fade_out
    right[:fade_samples] *= fade_in
    right[-fade_samples:] *= fade_out

    return np.column_stack([left, right])
