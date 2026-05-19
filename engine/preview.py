"""
engine/preview.py
-----------------
Real-time audio preview via sounddevice.
Supports finite and infinite (looping) modes.
"""

import sys
import signal
import warnings
from pathlib import Path

import numpy as np

from . import config
from .preset_loader import load_preset
from .ocean_engine import OceanEngine


class PreviewSession:
    """Stream ocean audio to speakers in real-time."""

    def __init__(
        self,
        preset_path: Path,
        infinite: bool = False,
        duration_override: float = None,
    ):
        self.preset_path = Path(preset_path)
        self.infinite = infinite
        self.duration_override = duration_override
        self._running = False

    def start(self):
        try:
            import sounddevice as sd
        except ImportError:
            sys.exit("Preview requires sounddevice: pip install sounddevice")

        # Load and build
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            preset = load_preset(self.preset_path)

        if self.duration_override:
            preset["duration"] = self.duration_override

        name = preset.get("meta", {}).get("name", self.preset_path.stem)
        duration = preset["duration"]

        print(f"🌊 Preview: {name}")
        print(f"   Duration: {'∞ (Ctrl+C to stop)' if self.infinite else f'{duration}s'}")
        print(f"   Building audio...")

        engine = OceanEngine(preset)
        audio = engine.build()

        print(f"   Streaming to speakers...")
        print()

        self._running = True
        signal.signal(signal.SIGINT, self._stop_handler)

        try:
            if self.infinite:
                while self._running:
                    sd.play(audio.astype(np.float32), config.SAMPLE_RATE)
                    sd.wait()
            else:
                sd.play(audio.astype(np.float32), config.SAMPLE_RATE)
                sd.wait()
        except Exception as e:
            print(f"\n   Error: {e}")
        finally:
            print("\n   ✓ Preview ended.")

    def _stop_handler(self, sig, frame):
        self._running = False
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
