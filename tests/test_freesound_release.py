import json
import tempfile
import unittest
import wave
from pathlib import Path

from render_freesound import (
    FREESOUND_LICENSE,
    MODELS_DIR,
    PACK_NAME,
    RELEASE_PRESETS,
    render_preset,
    write_release_metadata,
)


class FreesoundReleaseTests(unittest.TestCase):
    def test_release_covers_every_current_model_once(self):
        available = {path.name for path in MODELS_DIR.glob("*.npz")}
        configured = [preset.model for preset in RELEASE_PRESETS]

        self.assertEqual(set(configured), available)
        self.assertEqual(len(configured), len(set(configured)))
        self.assertEqual(
            [preset.display_name for preset in RELEASE_PRESETS],
            ["Calma", "Onda Lunga", "Profondo", "Tempesta"],
        )

    def test_chunked_render_writes_audio_and_upload_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            sound = render_preset(
                RELEASE_PRESETS[1],
                output_dir,
                duration=0.12,
                sample_rate=8_000,
                subtype="PCM_16",
                chunk_seconds=0.05,
                fade_seconds=0.02,
                overwrite=False,
                show_progress=False,
            )
            write_release_metadata(output_dir, [sound])

            audio_path = output_dir / sound["filename"]
            with wave.open(str(audio_path), "rb") as rendered:
                self.assertEqual(rendered.getframerate(), 8_000)
                self.assertEqual(rendered.getnchannels(), 2)
                self.assertEqual(rendered.getnframes(), 960)
                first_frame = rendered.readframes(1)
                rendered.setpos(rendered.getnframes() - 1)
                last_frame = rendered.readframes(1)
            self.assertEqual(first_frame, b"\x00\x00\x00\x00")
            self.assertEqual(last_frame, b"\x00\x00\x00\x00")

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["pack_name"], PACK_NAME)
            self.assertEqual(manifest["license"], FREESOUND_LICENSE)
            self.assertEqual(manifest["sounds"][0]["seed"], RELEASE_PRESETS[1].seed)
            self.assertTrue(manifest["sounds"][0]["sha256"])
            self.assertEqual(
                manifest["sounds"][0]["title"],
                "Rolling Ocean Waves – Long Relaxing Swells",
            )
            first_paragraph, maresono_paragraph = manifest["sounds"][0][
                "description"
            ].split("\n\n", 1)
            self.assertNotIn("Maresono", first_paragraph)
            self.assertTrue(maresono_paragraph.startswith("Created with Maresono"))
            self.assertIn("License: Creative Commons 0 (CC0).", (
                output_dir / "FREESOUND_UPLOAD.md"
            ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
