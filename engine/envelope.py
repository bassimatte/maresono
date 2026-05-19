"""
engine/envelope.py
------------------
Envelope generators for natural ocean wave dynamics.
"""

import numpy as np
from . import config


def wave_envelope(
    duration: float,
    cycle_period: float = 8.0,
    attack_ratio: float = 0.45,
    randomize: float = 0.3,
) -> np.ndarray:
    """
    Create amplitude envelope mimicking ocean wave cycles.
    Each cycle: slow rise (wave building) → faster fall (wave breaking).
    Cycle timing is randomized for organic feel.
    """
    n_samples = int(duration * config.SAMPLE_RATE)
    envelope = np.zeros(n_samples)

    pos = 0
    while pos < n_samples:
        # Randomize period within ±randomize range
        this_period = cycle_period * (1.0 - randomize + 2.0 * randomize * np.random.random())
        this_samples = int(this_period * config.SAMPLE_RATE)
        if pos + this_samples > n_samples:
            this_samples = n_samples - pos
        if this_samples <= 0:
            break

        # Asymmetric: slow attack, faster decay
        attack_len = int(this_samples * attack_ratio)
        decay_len = this_samples - attack_len

        rise = np.sin(np.linspace(0, np.pi / 2, max(attack_len, 1))) ** 2
        fall = np.cos(np.linspace(0, np.pi / 2, max(decay_len, 1))) ** 1.5

        # Per-cycle amplitude variation
        amplitude = 0.5 + 0.5 * np.random.random()
        cycle_env = np.concatenate([rise, fall]) * amplitude

        envelope[pos:pos + this_samples] = cycle_env[:this_samples]
        pos += this_samples

    return envelope


def risacca_envelope(
    duration: float,
    cycle_period: float = 6.0,
    delay_secs: float = 2.0,
) -> np.ndarray:
    """
    Envelope for undertow/risacca — shorter, sharper bursts
    delayed from the main wave.
    """
    n_samples = int(duration * config.SAMPLE_RATE)

    # Sharper attack (backwash is sudden)
    env = wave_envelope(duration, cycle_period=cycle_period, attack_ratio=0.2, randomize=0.35)

    # Shift to simulate delay after wave impact
    shift = int(delay_secs * config.SAMPLE_RATE)
    env = np.roll(env, shift)
    env[:shift] *= np.linspace(0, 1, shift)  # Fade-in the shifted portion

    return env


def crossfade_loop(audio: np.ndarray, fade_secs: float = 3.0) -> np.ndarray:
    """Apply crossfade at start/end for seamless looping."""
    fade_samples = int(fade_secs * config.SAMPLE_RATE)
    fade_samples = min(fade_samples, len(audio) // 4)

    fade_in = np.linspace(0, 1, fade_samples) ** 2
    fade_out = np.linspace(1, 0, fade_samples) ** 2

    if audio.ndim == 2:
        audio[:fade_samples] *= fade_in[:, np.newaxis]
        audio[-fade_samples:] *= fade_out[:, np.newaxis]
    else:
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

    return audio
