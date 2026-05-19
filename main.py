"""
main.py — Maresono V1.0
------------------------
The Sound of the Sea — Ocean wave synthesizer.

Usage:
    python main.py                              # process all presets
    python main.py --list                       # list available presets
    python main.py --preset path/to.yaml        # single preset by path
    python main.py --name "Calm"               # run preset by name
    python main.py --duration 300              # override duration (seconds)
    python main.py --format flac              # export as FLAC (wav, flac, ogg, mp3)
    python main.py --preview --name "Storm"   # real-time preview to speakers
    python main.py --preview --infinite       # infinite ocean, Ctrl+C to stop
    python main.py --generate                 # generate a random preset
    python main.py --generate --mood calm     # generate with mood bias
    python main.py --gui                      # launch web UI in browser
    python main.py --gui --host 0.0.0.0      # expose web UI for containers/LAN
"""

import argparse
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

if "--hires" in sys.argv:
    from engine.config import set_hires
    set_hires()

import numpy as np

from engine.preset_loader import load_preset
from engine.ocean_engine import OceanEngine
from engine.exporter import export_audio, SUPPORTED_FORMATS
from engine.generator import generate_preset, save_generated_preset, get_available_moods

PRESET_DIR = Path("presets")
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


# ── Progress bar ──────────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total: int, description: str = "", width: int = 40):
        self.total = max(total, 1)
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current = min(self.current + n, self.total)
        self._render()

    def _render(self):
        frac = self.current / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        percent = frac * 100
        elapsed = time.time() - self.start_time
        if self.current > 0 and frac < 1.0:
            eta = elapsed / frac * (1 - frac)
            time_str = f"ETA {eta:.0f}s"
        elif frac >= 1.0:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = "..."
        line = f"\r  {self.description} |{bar}| {percent:5.1f}% {time_str}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        self.current = self.total
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()


# ── Helpers ───────────────────────────────────────────────────────────────────

def discover_presets() -> List[Path]:
    if not PRESET_DIR.exists():
        return []
    return sorted(PRESET_DIR.rglob("*.yaml"))


def find_preset_by_name(preset_paths: List[Path], name: str) -> List[Path]:
    query = name.lower()
    matches = []
    for p in preset_paths:
        if query in p.stem.lower():
            matches.append(p)
            continue
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                preset = load_preset(p)
            meta_name = preset.get("meta", {}).get("name", "")
            if query in meta_name.lower():
                matches.append(p)
        except Exception:
            pass
    return matches


def list_presets(preset_paths: List[Path]):
    print(f"{'#':<4} {'Name':<30} {'Category':<14} {'Duration':<10} Path")
    print("─" * 90)
    for i, p in enumerate(preset_paths, 1):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                preset = load_preset(p)
            meta = preset.get("meta", {})
            name = meta.get("name", p.stem)
            cat = meta.get("category", "—")
            dur = f"{preset['duration']:.0f}s"
        except Exception:
            name, cat, dur = p.stem, "?", "?"
        print(f"{i:<4} {name:<30} {cat:<14} {dur:<10} {p}")
    print(f"\n{len(preset_paths)} preset(s) found.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(preset_paths: List[Path], cli_duration: Optional[float], audio_format: str):
    ok = failed = 0
    total = len(preset_paths)

    for idx, preset_path in enumerate(preset_paths, 1):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                preset = load_preset(preset_path)
                for w in caught:
                    print(f"  ⚠  {w.message}")

            if cli_duration is not None:
                preset["duration"] = cli_duration

            name = preset["meta"].get("name", preset_path.stem)
            slug = preset["meta"].get("slug", preset_path.stem.lower().replace(" ", "_"))
            print(f"\n[{idx}/{total}] 🌊 Generating: {name}")

            engine = OceanEngine(preset)
            n_layers = len([l for l in preset["layers"] if l.get("enabled", True)])
            has_binaural = bool(preset.get("binaural", {}).get("enabled"))
            has_reverb = bool(preset.get("reverb", {}).get("enabled"))
            total_steps = n_layers + int(has_binaural) + int(has_reverb) + 1

            progress = ProgressBar(total_steps, description="Building")
            audio = engine.build(progress_callback=progress.update)
            progress.finish()

            audio_filename = f"{slug}.{audio_format}"
            out_path = EXPORT_DIR / audio_filename
            export_audio(out_path, audio, fmt=audio_format)
            print(f"  ✓ Saved: {out_path}")
            ok += 1

        except Exception as exc:
            print(f"  ✗ Failed ({preset_path.name}): {exc}", file=sys.stderr)
            failed += 1

    print(f"\n{'─' * 40}")
    print(f"Done — {ok} exported, {failed} failed.")


def main():
    available_moods = get_available_moods()

    parser = argparse.ArgumentParser(description="Maresono V1.0 — The Sound of the Sea")
    parser.add_argument("--gui", action="store_true", help="Launch web UI in browser")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for web UI server")
    parser.add_argument("--list", action="store_true", help="List available presets")
    parser.add_argument("--preview", action="store_true", help="Real-time preview to speakers")
    parser.add_argument("--infinite", action="store_true", help="Infinite loop (with --preview)")
    parser.add_argument("--generate", action="store_true", help="Generate a random preset")
    parser.add_argument("--mood", type=str, default=None, choices=available_moods,
                       help="Mood bias for --generate")
    parser.add_argument("--generate-count", type=int, default=1, help="Number of presets to generate")
    parser.add_argument("--preset", type=Path, default=None, help="Path to a single preset (.yaml)")
    parser.add_argument("--name", type=str, default=None, help="Run preset(s) by name search")
    parser.add_argument("--duration", type=float, default=None, help="Override duration (seconds)")
    parser.add_argument("--format", type=str, default="wav", choices=SUPPORTED_FORMATS,
                       help="Audio export format")
    parser.add_argument("--hires", action="store_true", help="48kHz/24-bit mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.hires:
        from engine.config import set_hires
        set_hires()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # GUI mode
    if args.gui:
        try:
            from engine.web_server import launch_gui
        except ImportError as e:
            sys.exit(f"Web UI requires: pip install fastapi uvicorn[standard]\nError: {e}")
        launch_gui(host=args.host)
        return

    # Generate mode
    if args.generate:
        generated_dir = PRESET_DIR / "generated"
        print(f"Generating {args.generate_count} preset(s)"
              f"{f' (mood: {args.mood})' if args.mood else ''}...\n")
        for i in range(args.generate_count):
            preset = generate_preset(mood=args.mood)
            path = save_generated_preset(preset, generated_dir)
            name = preset["meta"]["name"]
            print(f"  {i+1}. {name} → {path}")
        print(f"\n✓ {args.generate_count} preset(s) saved.")
        return

    # Standard modes
    all_presets = discover_presets()

    if args.list:
        if not all_presets:
            sys.exit(f"No presets found in {PRESET_DIR}/")
        list_presets(all_presets)
        return

    # Determine which preset(s)
    if args.preset:
        if not args.preset.exists():
            sys.exit(f"Preset not found: {args.preset}")
        preset_paths = [args.preset]
    elif args.name:
        if not all_presets:
            sys.exit(f"No presets found in {PRESET_DIR}/")
        preset_paths = find_preset_by_name(all_presets, args.name)
        if not preset_paths:
            sys.exit(f"No preset matching '{args.name}'. Use --list to see available presets.")
        print(f"Matched {len(preset_paths)} preset(s) for '{args.name}'.\n")
    else:
        preset_paths = all_presets
        if not preset_paths:
            sys.exit(f"No presets found in {PRESET_DIR}/. Use --generate to create one.")
        print(f"Found {len(preset_paths)} preset(s).\n")

    # Preview mode
    if args.preview:
        from engine.preview import PreviewSession
        session = PreviewSession(
            preset_path=preset_paths[0],
            infinite=args.infinite,
            duration_override=args.duration,
        )
        session.start()
        return

    # Normal render
    run(preset_paths, cli_duration=args.duration, audio_format=args.format)


if __name__ == "__main__":
    main()
