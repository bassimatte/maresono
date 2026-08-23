import unittest
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


if __name__ == "__main__":
    unittest.main()
