"""
Maresono — Auto-Tuner

Takes a reference audio file (real ocean drum recording) and generates
a calibrated YAML preset that matches its spectral characteristics.

Usage:
    python auto_tune.py path/to/reference.wav --name "Model Name"
    python auto_tune.py --all   (processes all files in MUS_wave_simulator/)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import soundfile as sf
    import yaml
except ImportError:
    print("pip install soundfile PyYAML")
    sys.exit(1)


def load_audio(path: str) -> tuple:
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def analyze(audio: np.ndarray, sr: int) -> dict:
    """Extract spectral features from reference audio."""
    hop = int(sr * 0.05)
    win = hop * 2
    n_frames = (len(audio) - win) // hop
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    window = np.hanning(win)

    # Spectrogram
    spec = np.zeros((n_frames, len(freqs)))
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + win] * window
        spec[i] = np.abs(np.fft.rfft(frame))

    power = spec ** 2

    # Spectral centroid
    total_power = power.sum(axis=1) + 1e-10
    centroid = (power * freqs[np.newaxis, :]).sum(axis=1) / total_power
    avg_centroid = float(np.mean(centroid))

    # Band energies
    bands = {
        'sub_bass': (20, 80),
        'bass': (80, 250),
        'low_mid': (250, 800),
        'mid': (800, 2500),
        'high_mid': (2500, 5000),
        'high': (5000, 12000),
    }
    band_energy = {}
    total_e = 0
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        e = float(power[:, mask].sum())
        band_energy[name] = e
        total_e += e

    band_pct = {k: v / total_e for k, v in band_energy.items()}

    # Amplitude envelope for dynamics
    win_env = int(sr * 0.1)
    n_env = len(audio) // win_env
    env = np.array([np.sqrt(np.mean(audio[i*win_env:(i+1)*win_env]**2))
                    for i in range(n_env)])
    # Filter out silence (below -40 dB from peak)
    peak_env = np.max(env)
    active = env > peak_env * 0.01
    if active.sum() > 10:
        env_active = env[active]
        env_db = 20 * np.log10(env_active + 1e-10)
        dynamic_range = float(np.percentile(env_db, 95) - np.percentile(env_db, 5))
    else:
        dynamic_range = 10.0

    # Swell period via autocorrelation
    if active.sum() > 20:
        env_c = env_active - env_active.mean()
        corr = np.correlate(env_c, env_c, mode='full')
        corr = corr[len(corr)//2:]
        corr /= corr[0] + 1e-10
        fps = 1000.0 / 100.0  # 100ms windows
        min_p = int(1.5 * fps)
        max_p = int(15.0 * fps)
        search = corr[min_p:min(max_p, len(corr))]
        if len(search) > 0:
            swell_period = float((np.argmax(search) + min_p) / fps)
        else:
            swell_period = 2.0
    else:
        swell_period = 2.0

    return {
        'centroid': avg_centroid,
        'centroid_p10': float(np.percentile(centroid, 10)),
        'centroid_p90': float(np.percentile(centroid, 90)),
        'bands': band_pct,
        'dynamic_range_db': dynamic_range,
        'swell_period': swell_period,
        'duration': float(len(audio) / sr),
    }


def generate_preset(analysis: dict, name: str) -> dict:
    """Generate a YAML preset calibrated to the analysis."""
    centroid = analysis['centroid']
    bands = analysis['bands']
    dynamics = analysis['dynamic_range_db']
    swell = analysis['swell_period']

    # Determine character from spectral balance
    mid_ratio = bands['mid']
    himid_ratio = bands['high_mid']
    lowmid_ratio = bands['low_mid']

    # Large sphere resonance: drives the mid-frequency content
    # Map centroid to membrane resonance (large spheres are lower half)
    large_resonance = int(np.clip(centroid * 0.25, 400, 900))

    # Small sphere resonance: drives high-mid content
    small_resonance = int(np.clip(centroid * 1.4, 2500, 5500))

    # Mix balance: ratio of large to small sphere volume
    # More mid = more large sphere contribution
    large_mix = 1.0
    small_mix = round(float(np.clip(himid_ratio / (mid_ratio + 0.01) * 1.5, 0.3, 1.2)), 2)

    # Mass affects response speed
    # Lower centroid = heavier/slower large spheres (more dampening)
    large_mass = round(float(np.clip(3500 / centroid, 1.0, 2.5)), 1)
    # Small metallic always heavier per-size
    small_mass = round(float(np.clip(centroid / 3000, 0.5, 1.2)), 1)

    # Dynamics → proximity floor (less dynamic = higher floor)
    # 8 dB → floor 0.7, 20 dB → floor 0.4
    proximity_note = f"dynamic_range={dynamics:.0f}dB"

    # Scatter: more variable centroid range = more scatter
    centroid_range = analysis['centroid_p90'] - analysis['centroid_p10']
    scatter = round(float(np.clip(centroid_range / 3000, 0.2, 0.7)), 2)

    # Determine description
    if himid_ratio > 0.35:
        character = "bright and shimmering"
    elif mid_ratio > 0.50:
        character = "warm and enveloping"
    elif lowmid_ratio > 0.10:
        character = "deep and resonant"
    else:
        character = "balanced and natural"

    slug = name.lower().replace(' ', '_').replace('"', '').replace("'", '')

    preset = {
        'meta': {
            'name': name,
            'slug': slug,
            'category': 'instrument',
            'mood': ['meditative', 'natural'],
            'description': (
                f"Calibrated from real {name} recording. "
                f"Character: {character}. "
                f"Spectral centroid: {centroid:.0f} Hz, "
                f"swell period: {swell:.1f}s."
            ),
            'author': 'Maresono (auto-tuned)',
            'tags': ['ocean_drum', 'physics', 'calibrated'],
        },
        'duration': 60,
        'layers': [
            {
                'name': 'Large Spheres (Wave)',
                'type': 'large_spheres',
                'enabled': True,
                'mix': large_mix,
                'pan': -0.1,
                'tilt': {
                    'cycle_period': round(swell, 1),
                    'randomize': 0.25,
                },
                'spheres': {
                    'count': 10,
                    'membrane_resonance': large_resonance,
                    'mass': large_mass,
                    'diameter': 0.35,
                },
                'frequency': {
                    'high': int(centroid * 0.9),
                },
            },
            {
                'name': 'Small Spheres (Risacca)',
                'type': 'small_spheres',
                'enabled': True,
                'mix': small_mix,
                'pan': 0.1,
                'tilt': {
                    'cycle_period': round(swell, 1),
                    'randomize': 0.25,
                },
                'spheres': {
                    'count': 20,
                    'brightness': 0.7,
                    'scatter': scatter,
                    'membrane_resonance': small_resonance,
                    'mass': small_mass,
                    'diameter': 0.35,
                },
                'frequency': {
                    'low': int(centroid * 0.8),
                },
            },
        ],
        'binaural': {'enabled': False},
        'reverb': {'enabled': False},
    }

    return preset


def main():
    parser = argparse.ArgumentParser(description="Auto-tune Maresono preset from reference audio")
    parser.add_argument("audio_file", nargs='?', help="Path to reference WAV")
    parser.add_argument("--name", type=str, default=None, help="Preset name")
    parser.add_argument("--all", action="store_true", help="Process all WAVs in MUS_wave_simulator/")
    parser.add_argument("--output-dir", type=str, default="presets", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.all:
        wav_dir = Path(__file__).parent.parent / "MUS_wave_simulator"
        if not wav_dir.exists():
            print(f"Error: {wav_dir} not found")
            sys.exit(1)
        files = list(wav_dir.glob("*.wav"))
        if not files:
            print("No WAV files found")
            sys.exit(1)

        # Map filenames to nice names
        name_map = {
            'mediterraneo': 'Mediterraneo',
            'powerful-sea': 'Mediterraneo Powerful',
            'love': 'Love',
            'intuition': 'Intuition',
            'crystal': 'Crystal',
            'cala-luna': 'Cala Luna',
            'chakra': 'Chakra Drum',
        }

        for f in sorted(files):
            fname = f.stem.lower()
            # Find matching name
            preset_name = None
            for key, nice_name in name_map.items():
                if key in fname:
                    preset_name = nice_name
                    break
            if not preset_name:
                preset_name = f.stem[:40]

            print(f"\n{'='*50}")
            print(f"  Processing: {f.name}")
            print(f"  Preset: {preset_name}")

            audio, sr = load_audio(str(f))
            analysis = analyze(audio, sr)
            preset = generate_preset(analysis, preset_name)

            out_path = output_dir / f"{preset['meta']['slug']}.yaml"
            with open(out_path, 'w') as fh:
                yaml.dump(preset, fh, default_flow_style=False, sort_keys=False)

            print(f"  Centroid: {analysis['centroid']:.0f} Hz | "
                  f"Dynamics: {analysis['dynamic_range_db']:.1f} dB | "
                  f"Swell: {analysis['swell_period']:.1f}s")
            print(f"  -> {out_path}")

    elif args.audio_file:
        path = Path(args.audio_file)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)

        name = args.name or path.stem[:30]
        print(f"\n  Analyzing: {path.name}")
        audio, sr = load_audio(str(path))
        analysis = analyze(audio, sr)
        preset = generate_preset(analysis, name)

        out_path = output_dir / f"{preset['meta']['slug']}.yaml"
        with open(out_path, 'w') as fh:
            yaml.dump(preset, fh, default_flow_style=False, sort_keys=False)

        print(f"  Centroid: {analysis['centroid']:.0f} Hz")
        print(f"  Dynamics: {analysis['dynamic_range_db']:.1f} dB")
        print(f"  Swell: {analysis['swell_period']:.1f}s")
        print(f"  -> Preset saved to: {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
