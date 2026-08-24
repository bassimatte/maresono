"""Render the current Maresono spectral presets for a Freesound release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

from engine.spectral_model import SpectralModel, SpectralSynthesizer


REPO_DIR = Path(__file__).resolve().parent
MODELS_DIR = REPO_DIR / "models"
DEFAULT_OUTPUT_DIR = REPO_DIR / "exports" / "freesound" / "maresono-release"
PACK_NAME = "Ocean and Sea Waves Generative Soundscapes by MARESONO"
FREESOUND_LICENSE = "Creative Commons 0"


@dataclass(frozen=True)
class ReleasePreset:
    model: str
    slug: str
    display_name: str
    release_title: str
    character: str
    intensity: float
    seed: int
    tags: tuple[str, ...]


RELEASE_PRESETS = (
    ReleasePreset(
        model="Calma.npz",
        slug="calma",
        display_name="Calma",
        release_title="Gentle Ocean Waves – Calm Shore Ambience",
        character=(
            "Gentle waves rise and fall with soft foam, restrained movement, and delicate "
            "shoreline detail. The sound remains calm and spacious, with subtle variations "
            "that create a natural sense of continuous water movement. Suitable for relaxation, "
            "meditation, quiet backgrounds, and atmospheric sound design."
        ),
        intensity=0.12,
        seed=26082401,
        tags=("calm", "gentle", "shore", "lapping"),
    ),
    ReleasePreset(
        model="Onda Lunga.npz",
        slug="onda-lunga",
        display_name="Onda Lunga",
        release_title="Rolling Ocean Waves – Long Relaxing Swells",
        character=(
            "Slow ocean waves move in long, rolling cycles, gradually building and receding with "
            "an unhurried tidal rhythm. Broad swells and gentle changes in texture create a "
            "peaceful, immersive sea atmosphere without obvious repetition. Suitable for "
            "relaxation, meditation, sleep, ambient backgrounds, and sound design."
        ),
        intensity=0.25,
        seed=26082402,
        tags=("long-wave", "swell", "rolling", "meditative"),
    ),
    ReleasePreset(
        model="Profondo.npz",
        slug="profondo",
        display_name="Profondo",
        release_title="Deep Ocean Rumble – Dark Sea Ambience",
        character=(
            "A deep and shadowy ocean atmosphere with low-frequency movement, distant waves, and "
            "the slow pull of an underwater current. The sound has more weight than a typical "
            "shoreline ambience, combining dark rumbling textures with evolving layers of water. "
            "Suitable for cinematic atmospheres, underwater scenes, dark ambient music, "
            "installations, and sound design."
        ),
        intensity=0.42,
        seed=26082403,
        tags=("deep", "undertow", "rumble", "dark"),
    ),
    ReleasePreset(
        model="Tempesta.npz",
        slug="tempesta",
        display_name="Tempesta",
        release_title="Powerful Storm Waves – Rough Sea and Surf",
        character=(
            "Powerful waves surge and break with dense foam, forceful motion, and an energetic "
            "stereo image. The texture suggests rough open water and heavy storm surf, with "
            "continuous changes in intensity and no obvious repeating pattern. Suitable for "
            "dramatic ocean scenes, cinematic backgrounds, installations, games, and atmospheric "
            "sound design."
        ),
        intensity=0.82,
        seed=26082404,
        tags=("storm", "surf", "powerful", "rough-sea"),
    ),
)

COMMON_TAGS = (
    "maresono",
    "ocean",
    "waves",
    "ocean-drum",
    "generative",
    "soundscape",
    "ambient",
    "spectral-synthesis",
    "non-looping",
)


def _duration_label(duration: float) -> str:
    minutes = duration / 60.0
    if math.isclose(minutes, round(minutes), abs_tol=1e-9):
        return f"{int(round(minutes))}min"
    return f"{duration:g}s"


def _bit_depth_label(subtype: str) -> str:
    return subtype.removeprefix("PCM_").lower() + "-bit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_description(preset: ReleasePreset, duration: float,
                         sample_rate: int, subtype: str) -> str:
    return (
        f"{preset.character}\n\n"
        "Created with Maresono, Matteo Bassi’s generative ocean-sound instrument, using "
        "spectral characteristics learned from a handcrafted ocean drum. This is a newly "
        "synthesized, non-looping soundscape rather than a field recording. "
        f"{duration / 60:g} minutes, stereo, {sample_rate / 1000:g} "
        f"kHz/{_bit_depth_label(subtype)} WAV. "
        "https://bassimatte.github.io/maresono/"
    )


def render_preset(
    preset: ReleasePreset,
    output_dir: Path,
    *,
    duration: float,
    sample_rate: int,
    subtype: str,
    chunk_seconds: float,
    fade_seconds: float,
    overwrite: bool,
    show_progress: bool = True,
) -> dict:
    """Render one preset incrementally so long releases stay memory-safe."""
    model_path = MODELS_DIR / preset.model
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    duration_tag = _duration_label(duration)
    rate_tag = f"{sample_rate // 1000}k" if sample_rate % 1000 == 0 else str(sample_rate)
    filename = (
        f"maresono-{preset.slug}-{duration_tag}-{rate_tag}-"
        f"{_bit_depth_label(subtype)}.wav"
    )
    output_path = output_dir / filename
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output_path}")

    synthesizer = SpectralSynthesizer(
        SpectralModel.load(model_path), seed=preset.seed
    )
    total_frames = int(round(duration * sample_rate))
    chunk_frames = max(1, int(round(chunk_seconds * sample_rate)))
    fade_frames = min(total_frames // 2, int(round(fade_seconds * sample_rate)))
    written_frames = 0
    sum_squares = 0.0
    sample_count = 0
    peak = 0.0

    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=sample_rate,
        channels=2,
        format="WAV",
        subtype=subtype,
    ) as output:
        while written_frames < total_frames:
            frames = min(chunk_frames, total_frames - written_frames)
            audio = synthesizer.synthesize(
                frames / sample_rate,
                sr=sample_rate,
                stereo=True,
                intensity=preset.intensity,
                fast=False,
            )
            if len(audio) < frames:
                audio = np.pad(audio, ((0, frames - len(audio)), (0, 0)))
            elif len(audio) > frames:
                audio = audio[:frames]

            if fade_frames > 0:
                positions = written_frames + np.arange(frames)
                fade_in = np.clip(positions / fade_frames, 0.0, 1.0)
                fade_out = np.clip(
                    (total_frames - 1 - positions) / fade_frames, 0.0, 1.0
                )
                audio = audio * np.minimum(fade_in, fade_out)[:, np.newaxis]

            output.write(audio)
            audio64 = audio.astype(np.float64, copy=False)
            sum_squares += float(np.sum(np.square(audio64)))
            sample_count += audio64.size
            peak = max(peak, float(np.max(np.abs(audio64))))
            written_frames += frames
            progress = written_frames / total_frames * 100.0
            if show_progress:
                print(
                    f"\r  {preset.display_name:<12} {progress:6.1f}%",
                    end="",
                    flush=True,
                )
    if show_progress:
        print()

    rms = math.sqrt(sum_squares / max(1, sample_count))
    return {
        "title": preset.release_title,
        "filename": filename,
        "model": preset.model,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": 2,
        "subtype": subtype,
        "intensity": preset.intensity,
        "seed": preset.seed,
        "description": _release_description(preset, duration, sample_rate, subtype),
        "tags": list(COMMON_TAGS + preset.tags),
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-12)), 2),
        "sha256": _sha256(output_path),
    }


def write_release_metadata(output_dir: Path, sounds: list[dict]) -> None:
    manifest = {
        "pack_name": PACK_NAME,
        "creator": "Matteo Bassi",
        "project_url": "https://bassimatte.github.io/maresono/",
        "source_url": "https://github.com/bassimatte/maresono",
        "rendered_on": date.today().isoformat(),
        "license": FREESOUND_LICENSE,
        "license_note": "Released under Creative Commons 0 (CC0).",
        "sounds": sounds,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {PACK_NAME}",
        "",
        "License: Creative Commons 0 (CC0).",
        "",
    ]
    for sound in sounds:
        lines.extend([
            f"## {sound['title']}",
            "",
            f"File: `{sound['filename']}`",
            "",
            sound["description"],
            "",
            "Tags: " + " ".join(sound["tags"]),
            "",
        ])
    (output_dir / "FREESOUND_UPLOAD.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all current Maresono models for a Freesound pack."
    )
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--subtype", default="PCM_24")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--fade-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.sample_rate <= 0 or args.chunk_seconds <= 0:
        raise SystemExit("Duration, sample rate and chunk length must be positive.")
    if args.fade_seconds < 0:
        raise SystemExit("Fade length cannot be negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(RELEASE_PRESETS)} Maresono sounds to {output_dir}\n")
    sounds = [
        render_preset(
            preset,
            output_dir,
            duration=args.duration,
            sample_rate=args.sample_rate,
            subtype=args.subtype,
            chunk_seconds=args.chunk_seconds,
            fade_seconds=args.fade_seconds,
            overwrite=args.overwrite,
        )
        for preset in RELEASE_PRESETS
    ]
    write_release_metadata(output_dir, sounds)
    print(f"\nReady for Freesound: {output_dir}")


if __name__ == "__main__":
    main()
