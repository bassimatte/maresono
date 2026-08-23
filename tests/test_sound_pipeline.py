import unittest
import wave
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np

from engine import web_server
from engine.spectral_model import SpectralModel, SpectralSynthesizer


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class SoundPipelineTests(unittest.TestCase):
    def test_onda_lunga_uses_slower_learned_timing(self):
        calma = SpectralSynthesizer(
            SpectralModel.load(MODELS_DIR / "Calma.npz"), seed=1
        )
        onda_lunga = SpectralSynthesizer(
            SpectralModel.load(MODELS_DIR / "Onda Lunga.npz"), seed=1
        )

        self.assertGreater(
            onda_lunga._model_period_scale(),
            calma._model_period_scale(),
        )

    def test_mastering_targets_stable_rms_and_soft_ceiling(self):
        source = np.linspace(-2.0, 2.0, 20_000, dtype=np.float64)
        mastered = SpectralSynthesizer._master_output(source, intensity=0.25)

        self.assertAlmostEqual(SpectralSynthesizer._rms(mastered), 0.105, delta=0.01)
        self.assertLessEqual(float(np.max(np.abs(mastered))), 0.98)

    def test_preview_session_is_retry_safe_and_evolves(self):
        session_id = f"test-{uuid4().hex}"
        common = {
            "model": "Onda Lunga.npz",
            "intensity": 0.25,
            "preview_duration": 1.0,
            "session_id": session_id,
        }
        first = web_server.RenderRequest(**common, chunk_index=0)
        second = web_server.RenderRequest(**common, chunk_index=1)

        first_buffer, _ = web_server._render_package(first, preview=True)
        retry_buffer, _ = web_server._render_package(first, preview=True)
        second_buffer, _ = web_server._render_package(second, preview=True)

        self.assertEqual(first_buffer.getvalue(), retry_buffer.getvalue())
        self.assertNotEqual(first_buffer.getvalue(), second_buffer.getvalue())
        with wave.open(BytesIO(first_buffer.getvalue()), "rb") as preview_wav:
            self.assertEqual(preview_wav.getframerate(), 44_100)
            self.assertEqual(preview_wav.getnchannels(), 2)

    def test_synthesis_state_survives_chunk_boundaries(self):
        synthesizer = SpectralSynthesizer(
            SpectralModel.load(MODELS_DIR / "Onda Lunga.npz"), seed=42
        )

        first = synthesizer.synthesize(2.0, sr=8_000, intensity=0.25)
        first_filter_tail = synthesizer._fir_states["far_left:quiet"].copy()
        second = synthesizer.synthesize(2.0, sr=8_000, intensity=0.25)

        self.assertEqual(synthesizer._stream_sample_index, 32_000)
        self.assertGreaterEqual(len(synthesizer._wave_states), 4)
        self.assertEqual(set(synthesizer._pan_states), {"far", "near"})
        self.assertEqual(set(synthesizer._rumble_states), {"far_left", "far_right"})
        self.assertFalse(
            np.array_equal(first_filter_tail, synthesizer._fir_states["far_left:quiet"])
        )

        window = 800
        before_rms = SpectralSynthesizer._rms(first[-window:])
        after_rms = SpectralSynthesizer._rms(second[:window])
        boundary_ratio = max(before_rms, after_rms) / min(before_rms, after_rms)
        self.assertLess(boundary_ratio, 1.25)

    def test_wave_and_spatial_phases_do_not_restart_per_chunk(self):
        synthesizer = SpectralSynthesizer(
            SpectralModel.load(MODELS_DIR / "Onda Lunga.npz"), seed=7
        )
        synthesizer._reset_stream_state(1_000)

        synthesizer._stream_time = 0.0
        first_envelope = synthesizer._generate_multilayer_envelope(
            4.0, 1_000, intensity=0.25, state_key="continuity-test"
        )
        first_phase = synthesizer._last_wave_phase.copy()
        first_pan = synthesizer._generate_spatial_pan(4_000, 1_000, 0.25, "far")

        synthesizer._stream_time = 4.0
        second_envelope = synthesizer._generate_multilayer_envelope(
            4.0, 1_000, intensity=0.25, state_key="continuity-test"
        )
        second_phase = synthesizer._last_wave_phase.copy()
        second_pan = synthesizer._generate_spatial_pan(4_000, 1_000, 0.25, "far")

        self.assertAlmostEqual(first_envelope[-1], second_envelope[0], delta=0.005)
        self.assertAlmostEqual(first_phase[-1], second_phase[0], delta=0.005)
        self.assertAlmostEqual(first_pan[-1], second_pan[0], delta=0.005)

    def test_browser_schedules_stateful_chunks_edge_to_edge(self):
        html = Path("engine/static/index.html").read_text(encoding="utf-8")

        self.assertNotIn("CROSSFADE_SECONDS", html)
        self.assertIn(
            "const nextStart = Math.max(audioContext.currentTime + 0.05, scheduled.endTime);",
            html,
        )
        self.assertIn("gain.gain.setValueAtTime(1, startTime);", html)


if __name__ == "__main__":
    unittest.main()
