"""
Maresono — Reference Audio Analyzer

Analyzes a real ocean drum recording and extracts spectral/temporal
features to tune our physical model parameters.

Usage:
    python analyze_reference.py path/to/reference.wav [--plot]

Output:
    - Spectral centroid over time (brightness trajectory)
    - Frequency band energy distribution (low/mid/high)
    - Amplitude envelope statistics (swell period, dynamics)
    - Suggested parameter tuning for spheres.py
"""

import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("pip install soundfile")
    sys.exit(1)


def load_audio(path: str) -> tuple:
    """Load audio file, convert to mono float."""
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def compute_spectrogram(audio: np.ndarray, sr: int, hop_ms: float = 50):
    """Compute magnitude spectrogram with given hop size."""
    hop = int(sr * hop_ms / 1000)
    win = hop * 2
    n_frames = (len(audio) - win) // hop
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    
    spec = np.zeros((n_frames, len(freqs)))
    window = np.hanning(win)
    
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + win] * window
        spec[i] = np.abs(np.fft.rfft(frame))
    
    return spec, freqs, hop


def spectral_centroid(spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Compute spectral centroid per frame."""
    power = spec ** 2
    total = power.sum(axis=1) + 1e-10
    centroid = (power * freqs[np.newaxis, :]).sum(axis=1) / total
    return centroid


def band_energy(spec: np.ndarray, freqs: np.ndarray) -> dict:
    """Compute energy in frequency bands over time."""
    power = spec ** 2
    
    bands = {
        'sub_bass': (20, 80),      # rumble
        'bass': (80, 250),         # body of wave
        'low_mid': (250, 800),     # warmth
        'mid': (800, 2500),        # presence
        'high_mid': (2500, 5000),  # brilliance (small spheres)
        'high': (5000, 12000),     # air/shimmer
    }
    
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = power[:, mask].sum(axis=1)
    
    return result


def amplitude_envelope(audio: np.ndarray, sr: int, window_ms: float = 100):
    """RMS amplitude envelope."""
    win = int(sr * window_ms / 1000)
    n_frames = len(audio) // win
    env = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * win:(i + 1) * win]
        env[i] = np.sqrt(np.mean(frame ** 2))
    return env


def detect_swell_period(envelope: np.ndarray, sr: int, window_ms: float = 100):
    """Detect dominant swell/cycle period from amplitude envelope."""
    # Autocorrelation
    env_centered = envelope - envelope.mean()
    corr = np.correlate(env_centered, env_centered, mode='full')
    corr = corr[len(corr) // 2:]
    corr /= corr[0] + 1e-10
    
    # Find first peak after zero crossing
    fps = 1000.0 / window_ms  # frames per second
    min_period_frames = int(2.0 * fps)  # min 2 seconds
    max_period_frames = int(20.0 * fps)  # max 20 seconds
    
    search = corr[min_period_frames:max_period_frames]
    if len(search) > 0:
        peak_idx = np.argmax(search) + min_period_frames
        period_seconds = peak_idx / fps
        return period_seconds, corr
    return None, corr


def suggest_parameters(bands: dict, centroid: np.ndarray, swell_period: float,
                       envelope: np.ndarray) -> dict:
    """Based on analysis, suggest model parameters."""
    # Average spectral centroid
    avg_centroid = np.mean(centroid)
    
    # Band ratios
    total_energy = sum(b.sum() for b in bands.values())
    band_ratios = {k: v.sum() / total_energy for k, v in bands.items()}
    
    # Dynamic range
    env_db = 20 * np.log10(envelope + 1e-10)
    dynamic_range = np.percentile(env_db, 95) - np.percentile(env_db, 5)
    
    suggestions = {
        'tilt_envelope': {
            'cycle_period': round(swell_period, 1) if swell_period else 8.0,
            'note': f'Detected swell period: {swell_period:.1f}s' if swell_period else 'Could not detect'
        },
        'large_spheres': {
            'membrane_resonance': int(np.clip(avg_centroid * 0.4, 80, 300)),
            'note': f'Low-freq content dominates. Bass ratio: {band_ratios["bass"]:.2%}'
        },
        'small_spheres': {
            'membrane_resonance': int(np.clip(avg_centroid * 1.5, 1500, 5000)),
            'note': f'High-mid ratio: {band_ratios["high_mid"]:.2%}, centroid: {avg_centroid:.0f} Hz'
        },
        'mix_balance': {
            'low_to_high_ratio': (band_ratios['bass'] + band_ratios['sub_bass']) / 
                                 (band_ratios['high_mid'] + band_ratios['high'] + 1e-10),
            'note': 'Ratio of low to high energy — use to set layer volumes'
        },
        'dynamics': {
            'dynamic_range_db': round(dynamic_range, 1),
            'note': 'Higher = more contrast between swells and silence'
        }
    }
    
    return suggestions


def print_report(suggestions: dict, bands: dict, centroid: np.ndarray, 
                 duration: float):
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("  MARESONO — Reference Audio Analysis")
    print("=" * 60)
    print(f"\n  Duration: {duration:.1f}s")
    print(f"  Avg spectral centroid: {np.mean(centroid):.0f} Hz")
    print(f"  Centroid range: {np.percentile(centroid, 10):.0f} – "
          f"{np.percentile(centroid, 90):.0f} Hz")
    
    print("\n  Band Energy Distribution:")
    total = sum(b.sum() for b in bands.values())
    for name, energy in bands.items():
        pct = energy.sum() / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {name:12s} {pct:5.1f}% {bar}")
    
    print("\n  Suggested Parameters:")
    print("  " + "-" * 40)
    for section, params in suggestions.items():
        print(f"\n  [{section}]")
        for key, val in params.items():
            if key == 'note':
                print(f"    # {val}")
            else:
                print(f"    {key}: {val}")
    
    print("\n" + "=" * 60)


def save_profile(suggestions: dict, bands: dict, centroid: np.ndarray, 
                 output_path: str):
    """Save spectral profile as YAML for model tuning."""
    import yaml
    
    profile = {
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_centroid_p10': float(np.percentile(centroid, 10)),
        'spectral_centroid_p90': float(np.percentile(centroid, 90)),
        'band_energy': {},
        'suggested_params': {}
    }
    
    total = sum(b.sum() for b in bands.values())
    for name, energy in bands.items():
        profile['band_energy'][name] = float(energy.sum() / total)
    
    for section, params in suggestions.items():
        profile['suggested_params'][section] = {
            k: v for k, v in params.items() if k != 'note'
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(profile, f, default_flow_style=False)
    
    print(f"\n  Profile saved to: {output_path}")


def plot_analysis(audio, sr, spec, freqs, hop, centroid, bands, envelope):
    """Optional matplotlib visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (Install matplotlib for plots: pip install matplotlib)")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    
    # Waveform
    t = np.arange(len(audio)) / sr
    axes[0].plot(t, audio, linewidth=0.3)
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")
    
    # Spectrogram
    time_axis = np.arange(spec.shape[0]) * hop / sr
    axes[1].pcolormesh(time_axis, freqs[:500], 
                       20 * np.log10(spec[:, :500].T + 1e-10),
                       shading='auto', cmap='magma')
    axes[1].set_title("Spectrogram (0-5kHz)")
    axes[1].set_ylabel("Frequency (Hz)")
    
    # Spectral centroid
    axes[2].plot(time_axis, centroid, color='orange')
    axes[2].set_title("Spectral Centroid")
    axes[2].set_ylabel("Hz")
    axes[2].set_ylim(0, 5000)
    
    # Band energies
    for name, energy in bands.items():
        axes[3].plot(time_axis, energy / energy.max(), label=name, alpha=0.7)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].set_title("Band Energies (normalized)")
    axes[3].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig("reference_analysis.png", dpi=150)
    print("  Plot saved to: reference_analysis.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze reference audio for Maresono tuning")
    parser.add_argument("audio_file", help="Path to reference WAV/FLAC/OGG file")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots")
    parser.add_argument("--save-profile", type=str, default=None,
                        help="Save spectral profile YAML (default: auto)")
    args = parser.parse_args()
    
    path = Path(args.audio_file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)
    
    print(f"\n  Loading: {path}")
    audio, sr = load_audio(str(path))
    duration = len(audio) / sr
    print(f"  Sample rate: {sr} Hz, Duration: {duration:.1f}s")
    
    print("  Computing spectrogram...")
    spec, freqs, hop = compute_spectrogram(audio, sr)
    
    print("  Analyzing spectral features...")
    centroid = spectral_centroid(spec, freqs)
    bands = band_energy(spec, freqs)
    
    print("  Computing amplitude envelope...")
    envelope = amplitude_envelope(audio, sr)
    
    print("  Detecting swell period...")
    swell_period, _ = detect_swell_period(envelope, sr)
    
    suggestions = suggest_parameters(bands, centroid, swell_period, envelope)
    
    print_report(suggestions, bands, centroid, duration)
    
    # Save profile
    profile_path = args.save_profile or str(path.stem) + "_profile.yaml"
    save_profile(suggestions, bands, centroid, profile_path)
    
    if args.plot:
        plot_analysis(audio, sr, spec, freqs, hop, centroid, bands, envelope)


if __name__ == "__main__":
    main()
