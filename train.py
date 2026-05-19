"""
train.py — Pre-compute spectral models from reference WAV recordings.

Usage:
    python train.py [--input ../reference_recordings] [--output models/]

After running, the WAV files are no longer needed at runtime.
"""

import argparse
from pathlib import Path

import soundfile as sf

from engine.spectral_model import SpectralModel


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Maresono spectral models")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../reference_recordings"),
        help="Directory with reference WAV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory for .npz model files",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(args.input.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {args.input}")
        return

    print(f"Found {len(wav_files)} recordings to process\n")

    for wav_path in wav_files:
        print(f"Processing: {wav_path.name}...")
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        model = SpectralModel.from_audio(audio, sr)

        out_path = args.output / f"{wav_path.stem}.npz"
        model.save(out_path)

        size_kb = out_path.stat().st_size / 1024
        print(f"  → {out_path.name} ({size_kb:.1f} KB)")

    print(f"\nDone! {len(wav_files)} models saved to {args.output}/")
    print("WAV files are no longer needed at runtime.")


if __name__ == "__main__":
    main()
