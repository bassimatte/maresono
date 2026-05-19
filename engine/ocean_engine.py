"""
engine/ocean_engine.py
----------------------
Core synthesis engine for Maresono.

Builds ocean sound from preset configuration:
  - Wave layer (swell): deep, enveloping, large sphere rolling
  - Risacca layer (undertow): textured, brilliant, metallic sphere layer
  - Optional: foam, distant waves, crystal resonance
  - Binaural theta beats
  - Convolution reverb (beach/cove/underwater)
  - Stereo spatial mix

V1.1: Spectral evolution + multi-wave overlap for realism
"""

import numpy as np

from . import config
from .noise import pink_noise, brown_noise, white_noise, noise_blend
from .filters import Filters
from .envelope import wave_envelope, risacca_envelope, crossfade_loop
from .binaural import generate_theta_modulation, generate_binaural_carrier
from .reverb import apply_reverb
from .spheres import LargeSphereLayer, SmallSphereLayer, generate_tilt_envelope


class OceanEngine:
    """Builds a complete ocean soundscape from a preset dict."""

    def __init__(self, preset: dict):
        self.preset = preset
        self.duration = float(preset.get("duration", 60))
        self.n_samples = int(self.duration * config.SAMPLE_RATE)

    def build(self, progress_callback=None) -> np.ndarray:
        """Build the full ocean audio. Returns stereo (N, 2) array."""
        layers = self.preset.get("layers", [])
        stereo_bus = np.zeros((self.n_samples, 2))

        # Build each layer
        for layer_cfg in layers:
            if not layer_cfg.get("enabled", True):
                continue

            layer_type = layer_cfg.get("type", "wave")
            if layer_type == "wave":
                audio = self._build_wave_layer(layer_cfg)
            elif layer_type == "risacca":
                audio = self._build_risacca_layer(layer_cfg)
            elif layer_type == "foam":
                audio = self._build_foam_layer(layer_cfg)
            elif layer_type == "distant":
                audio = self._build_distant_layer(layer_cfg)
            elif layer_type == "crystal":
                audio = self._build_crystal_layer(layer_cfg)
            elif layer_type == "large_spheres":
                audio = self._build_large_spheres(layer_cfg)
            elif layer_type == "small_spheres":
                audio = self._build_small_spheres(layer_cfg)
            elif layer_type == "spectral_learned":
                audio = self._build_spectral_learned(layer_cfg)
            else:
                audio = self._build_wave_layer(layer_cfg)

            # audio is already stereo (N, 2) from multi-wave builders
            if audio.ndim == 1:
                mix = float(layer_cfg.get("mix", 1.0))
                pan = float(layer_cfg.get("pan", 0.0))
                stereo = self._pan_mono_to_stereo(audio * mix, pan)
            else:
                stereo = audio * float(layer_cfg.get("mix", 1.0))

            stereo_bus += stereo

            if progress_callback:
                progress_callback()

        # Binaural beats
        binaural_cfg = self.preset.get("binaural", {})
        if binaural_cfg.get("enabled", False):
            stereo_bus = self._apply_binaural(stereo_bus, binaural_cfg)
            if progress_callback:
                progress_callback()

        # Reverb
        reverb_cfg = self.preset.get("reverb", {})
        if reverb_cfg.get("enabled", False):
            stereo_bus = apply_reverb(
                stereo_bus,
                space=reverb_cfg.get("space", "beach"),
                mix=float(reverb_cfg.get("mix", 0.25)),
                decay_trim=float(reverb_cfg.get("decay_trim", 1.0)),
            )
            if progress_callback:
                progress_callback()

        # Crossfade for seamless looping
        stereo_bus = crossfade_loop(stereo_bus, fade_secs=config.FADE_SECS)

        # Final normalization
        peak = np.max(np.abs(stereo_bus))
        if peak > 0:
            stereo_bus = stereo_bus / peak * 0.9

        if progress_callback:
            progress_callback()

        return stereo_bus

    # ── Layer builders ────────────────────────────────────────────────────────

    def _build_wave_layer(self, cfg: dict) -> np.ndarray:
        """
        Main ocean swell with spectral evolution and multi-wave overlap.

        Realism approach:
        - Multiple independent wave events overlapping at random phases
        - Each wave evolves spectrally: lows build first → highs burst at crest → mids recede
        - Stereo sweep: each wave pans slightly as it passes
        """
        n_waves = int(cfg.get("wave_count", 4))
        env_cfg = cfg.get("envelope", {})
        cycle_period = float(env_cfg.get("cycle_period", 8.0))
        randomize = float(env_cfg.get("randomize", 0.3))
        freq = cfg.get("frequency", {})
        noise_cfg = cfg.get("noise", {})

        stereo_out = np.zeros((self.n_samples, 2))

        for i in range(n_waves):
            # Each wave has a random phase offset so they overlap naturally
            phase_offset = np.random.uniform(0, cycle_period)
            # Slight variation in period per wave
            wave_period = cycle_period * (1.0 + np.random.uniform(-0.2, 0.2))
            # Random amplitude weight (some waves bigger than others)
            wave_amp = np.random.uniform(0.5, 1.0)
            # Random pan position that drifts (wave moves across listener)
            pan_center = np.random.uniform(-0.4, 0.4)
            pan_drift = np.random.uniform(-0.3, 0.3)

            mono = self._build_single_wave(
                wave_period=wave_period,
                phase_offset=phase_offset,
                randomize=randomize,
                freq_cfg=freq,
                noise_cfg=noise_cfg,
            )

            # Animated stereo: pan sweeps during each wave cycle
            pan_envelope = self._animated_pan(pan_center, pan_drift, wave_period)
            stereo_wave = self._pan_mono_animated(mono * wave_amp, pan_envelope)
            stereo_out += stereo_wave

        # Normalize multi-wave sum
        peak = np.max(np.abs(stereo_out))
        if peak > 0:
            stereo_out /= peak

        return stereo_out

    def _build_single_wave(
        self,
        wave_period: float,
        phase_offset: float,
        randomize: float,
        freq_cfg: dict,
        noise_cfg: dict,
    ) -> np.ndarray:
        """
        Build a single wave event with spectral evolution.
        The spectrum changes through the wave cycle:
          - Build phase: dominated by low frequencies (swell approaching)
          - Crest/break: high frequencies burst in (white water, spray)
          - Recede: mid frequencies (water pulling back)
        """
        low = float(freq_cfg.get("low", 60))
        high = float(freq_cfg.get("high", 1200))

        # Generate three noise bands
        noise_low = brown_noise(self.n_samples)
        noise_low = Filters.bandpass(noise_low, low, min(low * 4, 400), order=3)

        noise_mid = pink_noise(self.n_samples)
        noise_mid = Filters.bandpass(noise_mid, 200, 1200, order=2)

        noise_high = white_noise(self.n_samples)
        noise_high = Filters.bandpass(noise_high, 800, high, order=2)

        # Create per-band envelopes with spectral timing
        # Low: builds first (attack-heavy)
        env_low = self._phase_shifted_envelope(
            wave_period, phase_offset, randomize,
            attack_ratio=0.55, peak_position=0.4
        )
        # High: peaks at crest (short burst)
        env_high = self._phase_shifted_envelope(
            wave_period, phase_offset + wave_period * 0.15, randomize * 0.8,
            attack_ratio=0.3, peak_position=0.5
        )
        # Clip high env to make it more impulsive (only the peaks)
        env_high = np.clip(env_high - 0.3, 0, 1.0) * (1.0 / 0.7)

        # Mid: peaks slightly after crest (receding water)
        env_mid = self._phase_shifted_envelope(
            wave_period, phase_offset + wave_period * 0.25, randomize,
            attack_ratio=0.35, peak_position=0.45
        )

        # Mix bands with spectral evolution
        signal = (
            noise_low * env_low * 0.6 +
            noise_mid * env_mid * 0.3 +
            noise_high * env_high * 0.2
        )

        return signal

    def _phase_shifted_envelope(
        self,
        cycle_period: float,
        phase_offset: float,
        randomize: float,
        attack_ratio: float = 0.45,
        peak_position: float = 0.45,
    ) -> np.ndarray:
        """Generate envelope with a time offset (phase shift)."""
        # Generate base envelope
        env = wave_envelope(
            self.duration,
            cycle_period=cycle_period,
            attack_ratio=attack_ratio,
            randomize=randomize,
        )
        # Apply phase offset by rolling
        shift_samples = int(phase_offset * config.SAMPLE_RATE)
        env = np.roll(env, shift_samples)
        # Smooth the wrap-around
        if shift_samples > 0:
            fade = min(shift_samples, int(0.5 * config.SAMPLE_RATE))
            env[:fade] *= np.linspace(0, 1, fade)
        return env

    def _animated_pan(self, center: float, drift: float, period: float) -> np.ndarray:
        """Create a slowly drifting pan position over time."""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        # Slow sinusoidal drift
        pan = center + drift * np.sin(2 * np.pi * t / period)
        return np.clip(pan, -1.0, 1.0)

    def _pan_mono_animated(self, mono: np.ndarray, pan: np.ndarray) -> np.ndarray:
        """Pan mono signal with time-varying pan position."""
        angle = (pan + 1.0) * np.pi / 4.0
        left = mono * np.cos(angle)
        right = mono * np.sin(angle)
        return np.column_stack([left, right])

    def _build_risacca_layer(self, cfg: dict) -> np.ndarray:
        """
        Undertow/backwash with multi-event overlap.
        Each risacca event is a burst of high-frequency textured noise,
        delayed from the main wave.
        """
        n_events = int(cfg.get("wave_count", 3))
        env_cfg = cfg.get("envelope", {})
        cycle_period = float(env_cfg.get("cycle_period", 6.0))
        delay = float(env_cfg.get("delay", 2.0))
        noise_cfg = cfg.get("noise", {})
        freq = cfg.get("frequency", {})
        low = float(freq.get("low", 500))
        high = float(freq.get("high", 6000))

        stereo_out = np.zeros((self.n_samples, 2))

        for i in range(n_events):
            # Independent noise for each event
            base = noise_blend(
                self.n_samples,
                white_mix=float(noise_cfg.get("white", 0.2)),
                pink_mix=float(noise_cfg.get("pink", 0.5)),
                brown_mix=float(noise_cfg.get("brown", 0.3)),
            )
            base = Filters.bandpass(base, low, high, order=int(freq.get("order", 3)))

            # Each risacca at different phase
            phase_offset = delay + np.random.uniform(-0.5, 1.0)
            event_period = cycle_period * (1.0 + np.random.uniform(-0.15, 0.15))

            env = self._phase_shifted_envelope(
                event_period, phase_offset, randomize=0.35,
                attack_ratio=0.2, peak_position=0.3
            )

            amp = np.random.uniform(0.5, 1.0)
            pan_center = np.random.uniform(0.0, 0.5)
            pan_drift = np.random.uniform(-0.2, 0.2)
            pan_env = self._animated_pan(pan_center, pan_drift, event_period)

            stereo_out += self._pan_mono_animated(base * env * amp, pan_env)

        peak = np.max(np.abs(stereo_out))
        if peak > 0:
            stereo_out /= peak

        return stereo_out

    def _build_foam_layer(self, cfg: dict) -> np.ndarray:
        """High-frequency foam/spray texture overlaid on wave crests."""
        base = white_noise(self.n_samples)

        freq = cfg.get("frequency", {})
        low = float(freq.get("low", 3000))
        high = float(freq.get("high", 12000))
        base = Filters.bandpass(base, low, high, order=2)

        env_cfg = cfg.get("envelope", {})
        env = wave_envelope(
            self.duration,
            cycle_period=float(env_cfg.get("cycle_period", 7.0)),
            attack_ratio=0.6,
            randomize=0.4,
        )
        # Foam only at peaks (threshold)
        env = np.clip(env - 0.3, 0, 1) * (1.0 / 0.7)

        return base * env

    def _build_distant_layer(self, cfg: dict) -> np.ndarray:
        """Distant waves — low rumble with very slow cycles."""
        base = brown_noise(self.n_samples)
        base = Filters.lowpass(base, float(cfg.get("frequency", {}).get("high", 300)), order=3)

        env = wave_envelope(
            self.duration,
            cycle_period=float(cfg.get("envelope", {}).get("cycle_period", 15.0)),
            attack_ratio=0.5,
            randomize=0.2,
        )

        return base * env

    def _build_crystal_layer(self, cfg: dict) -> np.ndarray:
        """
        Crystal resonance — pitched harmonic shimmer.
        Inspired by quartz crystal ocean drums.
        """
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        freq = float(cfg.get("frequency", {}).get("fundamental", 528))
        harmonics = int(cfg.get("harmonics", 5))
        decay = float(cfg.get("harmonic_decay", 0.6))

        signal = np.zeros(self.n_samples)
        for h in range(1, harmonics + 1):
            amp = decay ** (h - 1)
            detune = 1.0 + np.random.uniform(-0.002, 0.002)
            signal += amp * np.sin(2 * np.pi * freq * h * detune * t)

        env = wave_envelope(
            self.duration,
            cycle_period=float(cfg.get("envelope", {}).get("cycle_period", 12.0)),
            attack_ratio=0.5,
            randomize=0.15,
        )
        env = 0.2 + 0.8 * env

        signal *= env
        signal /= np.max(np.abs(signal)) + 1e-10
        return signal

    def _build_large_spheres(self, cfg: dict) -> np.ndarray:
        """
        Large sphere layer — physical simulation.
        Heavy glass/plastic spheres rolling collectively → deep swell.
        Uses shared tilt from preset (stored for sync with small spheres).
        """
        tilt_cfg = cfg.get("tilt", {})
        tilt = self._get_shared_tilt(tilt_cfg)

        sphere_cfg = cfg.get("spheres", {})
        layer = LargeSphereLayer(sphere_cfg, self.duration)
        mono = layer.build(tilt)

        # Apply some low-pass to keep it deep/soft
        cutoff = float(cfg.get("frequency", {}).get("high", 1500))
        mono = Filters.lowpass(mono, cutoff, order=2)

        return mono

    def _build_small_spheres(self, cfg: dict) -> np.ndarray:
        """
        Small sphere layer — physical simulation.
        Many light metallic spheres → bright shimmer.
        Uses SAME tilt as large spheres (same physical object).
        """
        tilt_cfg = cfg.get("tilt", {})
        tilt = self._get_shared_tilt(tilt_cfg)

        sphere_cfg = cfg.get("spheres", {})
        layer = SmallSphereLayer(sphere_cfg, self.duration)
        mono = layer.build(tilt)

        # High-pass to keep only the bright metallic content
        cutoff = float(cfg.get("frequency", {}).get("low", 1000))
        mono = Filters.highpass(mono, cutoff, order=2)

        return mono

    def _build_spectral_learned(self, cfg: dict) -> np.ndarray:
        """
        Learned spectral model — synthesizes audio that matches a real recording.
        Reads a reference WAV, learns its spectral envelope and dynamics,
        then generates new audio with identical character.
        """
        from .spectral_model import learn_from_file, SpectralSynthesizer
        import os

        reference = cfg.get("reference", "")
        # Resolve relative paths from project root
        if not os.path.isabs(reference):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            reference = os.path.join(base_dir, reference)

        if not os.path.exists(reference):
            print(f"  Warning: reference file not found: {reference}")
            return np.zeros(self.n_samples)

        model = learn_from_file(reference)
        synth = SpectralSynthesizer(model)
        mono = synth.synthesize(self.duration)

        return mono

    def _get_shared_tilt(self, tilt_cfg: dict) -> np.ndarray:
        """Get or create the shared tilt envelope (same for all sphere layers)."""
        if not hasattr(self, '_shared_tilt'):
            self._shared_tilt = generate_tilt_envelope(
                self.duration,
                cycle_period=float(tilt_cfg.get("cycle_period", 8.0)),
                randomize=float(tilt_cfg.get("randomize", 0.3)),
            )
        return self._shared_tilt

    # ── Spatial & effects ─────────────────────────────────────────────────────

    def _pan_mono_to_stereo(self, mono: np.ndarray, pan: float) -> np.ndarray:
        """Pan mono signal to stereo. pan: -1 (left) to +1 (right)."""
        angle = (pan + 1.0) * np.pi / 4.0
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)
        return np.column_stack([mono * left_gain, mono * right_gain])

    def _apply_binaural(self, stereo: np.ndarray, cfg: dict) -> np.ndarray:
        """Apply binaural theta beats to the stereo mix."""
        method = cfg.get("method", "both")
        beat_hz = float(cfg.get("beat_hz", 6.0))

        if method in ("embedded", "both"):
            depth = float(cfg.get("mod_depth", 0.15))
            mod = generate_theta_modulation(self.n_samples, beat_hz, depth)
            stereo[:, 0] *= mod
            stereo[:, 1] *= np.roll(mod, int(0.5 / beat_hz * config.SAMPLE_RATE))

        if method in ("carrier", "both"):
            carrier = generate_binaural_carrier(
                self.duration,
                carrier_hz=float(cfg.get("carrier_hz", 150.0)),
                beat_hz=beat_hz,
                amplitude=float(cfg.get("carrier_amplitude", 0.06)),
            )
            stereo += carrier

        return stereo
