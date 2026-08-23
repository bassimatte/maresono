"""
engine/spectral_model.py
------------------------
Analysis-resynthesis approach: learn directly from real ocean drum recordings.

Instead of simulating physics, we:
1. Extract the SPECTRAL ENVELOPE (frequency shape) from the real recording
2. Extract the AMPLITUDE MODULATION pattern (how volume swells/fades over time)
3. Generate new audio by shaping noise with those real characteristics

This produces sound that is spectrally identical to the real instrument,
with natural wave-like dynamics, but infinitely generatable (not a loop).
"""

from pathlib import Path

import numpy as np
from scipy.signal import stft, butter, sosfilt
from scipy.interpolate import interp1d
from . import config


class SpectralModel:
    """
    Learned spectral model from a real ocean drum recording.
    
    Stores:
    - avg_spectrum: the average spectral shape (what frequencies are present)
    - amp_envelope_stats: statistics of the amplitude modulation
    - spectral_flux: how the spectrum changes over time (brightness variation)
    """

    def __init__(self, avg_spectrum: np.ndarray, freqs: np.ndarray,
                 amp_stats: dict, spectral_flux: dict):
        self.avg_spectrum = avg_spectrum
        self.freqs = freqs
        self.amp_stats = amp_stats
        self.spectral_flux = spectral_flux

    @classmethod
    def from_audio(cls, audio: np.ndarray, sr: int) -> 'SpectralModel':
        """Learn a spectral model from a real recording."""
        # Compute STFT
        nperseg = 4096
        f, t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=nperseg // 2)

        magnitude = np.abs(Zxx)

        # 1. Average spectral envelope (the "timbre fingerprint")
        avg_spectrum = np.mean(magnitude, axis=1)
        # Smooth it to get the envelope shape (not individual harmonics)
        from scipy.ndimage import uniform_filter1d
        avg_spectrum_smooth = uniform_filter1d(avg_spectrum, size=8)

        # 2. Amplitude envelope over time
        rms_per_frame = np.sqrt(np.mean(magnitude ** 2, axis=0))
        # Normalize
        rms_norm = rms_per_frame / (np.max(rms_per_frame) + 1e-10)

        # Extract amplitude statistics
        amp_stats = {
            'mean': float(np.mean(rms_norm)),
            'std': float(np.std(rms_norm)),
            'min': float(np.percentile(rms_norm, 5)),
            'max': float(np.percentile(rms_norm, 95)),
            # Autocorrelation for periodicity detection
            'period_seconds': cls._detect_period(rms_norm, len(t) / (len(audio) / sr)),
            # The actual envelope pattern (downsampled for storage)
            'envelope_pattern': cls._extract_envelope_pattern(rms_norm, sr, len(audio), len(t)),
        }

        # 3. Spectral flux — how brightness changes with amplitude
        centroid_per_frame = np.sum(f[:, np.newaxis] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        spectral_flux = {
            'centroid_mean': float(np.mean(centroid_per_frame)),
            'centroid_std': float(np.std(centroid_per_frame)),
            'centroid_vs_amplitude': cls._correlation(rms_norm, centroid_per_frame),
            # Spectral variation over time (bright when loud, dark when quiet?)
            'bright_spectrum': None,  # filled below
            'quiet_spectrum': None,   # filled below
        }

        # Separate spectra for loud vs quiet moments
        loud_mask = rms_norm > np.percentile(rms_norm, 70)
        quiet_mask = rms_norm < np.percentile(rms_norm, 30)
        if loud_mask.sum() > 3:
            spectral_flux['bright_spectrum'] = np.mean(magnitude[:, loud_mask], axis=1)
        if quiet_mask.sum() > 3:
            spectral_flux['quiet_spectrum'] = np.mean(magnitude[:, quiet_mask], axis=1)

        return cls(avg_spectrum_smooth, f, amp_stats, spectral_flux)

    def save(self, path: Path):
        """Save pre-computed model to .npz file."""
        data = {
            'avg_spectrum': self.avg_spectrum,
            'freqs': self.freqs,
            'amp_mean': self.amp_stats['mean'],
            'amp_std': self.amp_stats['std'],
            'amp_min': self.amp_stats['min'],
            'amp_max': self.amp_stats['max'],
            'amp_period_seconds': self.amp_stats['period_seconds'],
            'amp_envelope_pattern': self.amp_stats['envelope_pattern'],
            'centroid_mean': self.spectral_flux['centroid_mean'],
            'centroid_std': self.spectral_flux['centroid_std'],
            'centroid_vs_amplitude': self.spectral_flux['centroid_vs_amplitude'],
        }
        if self.spectral_flux.get('bright_spectrum') is not None:
            data['bright_spectrum'] = self.spectral_flux['bright_spectrum']
        if self.spectral_flux.get('quiet_spectrum') is not None:
            data['quiet_spectrum'] = self.spectral_flux['quiet_spectrum']
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: Path) -> 'SpectralModel':
        """Load a pre-computed model from .npz file."""
        with np.load(path, allow_pickle=False) as data:
            avg_spectrum = data['avg_spectrum']
            freqs = data['freqs']
            amp_stats = {
                'mean': float(data['amp_mean']),
                'std': float(data['amp_std']),
                'min': float(data['amp_min']),
                'max': float(data['amp_max']),
                'period_seconds': float(data['amp_period_seconds']),
                'envelope_pattern': data['amp_envelope_pattern'],
            }
            spectral_flux = {
                'centroid_mean': float(data['centroid_mean']),
                'centroid_std': float(data['centroid_std']),
                'centroid_vs_amplitude': float(data['centroid_vs_amplitude']),
                'bright_spectrum': data['bright_spectrum'] if 'bright_spectrum' in data else None,
                'quiet_spectrum': data['quiet_spectrum'] if 'quiet_spectrum' in data else None,
            }
        return cls(avg_spectrum, freqs, amp_stats, spectral_flux)

    @staticmethod
    def _detect_period(envelope: np.ndarray, fps: float) -> float:
        """Detect dominant period from amplitude envelope."""
        env_centered = envelope - np.mean(envelope)
        corr = np.correlate(env_centered, env_centered, mode='full')
        corr = corr[len(corr) // 2:]
        corr /= corr[0] + 1e-10

        # Search between 1 and 10 seconds
        min_frames = int(1.0 * fps)
        max_frames = int(10.0 * fps)
        search = corr[min_frames:min(max_frames, len(corr))]
        if len(search) > 0:
            peak = np.argmax(search) + min_frames
            return float(peak / fps)
        return 2.0

    @staticmethod
    def _extract_envelope_pattern(rms_norm: np.ndarray, sr: int,
                                   n_audio_samples: int, n_frames: int) -> np.ndarray:
        """Extract a representative envelope pattern (one full cycle)."""
        # Downsample to ~10 Hz for compact storage
        target_len = min(200, len(rms_norm))
        indices = np.linspace(0, len(rms_norm) - 1, target_len).astype(int)
        return rms_norm[indices]

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation between two signals."""
        a_c = a - np.mean(a)
        b_c = b - np.mean(b)
        denom = (np.std(a) * np.std(b) + 1e-10)
        return float(np.mean(a_c * b_c) / denom)


class SpectralSynthesizer:
    """
    V2.0 — Generates realistic ocean wave audio from a learned SpectralModel.
    
    Features:
    - FIR filter convolution (smooth, no frame artifacts)
    - Multi-layer envelope (slow swells + individual waves)
    - Stereo movement (offset envelopes between L/R)
    - Receding wash (different spectral character for decay phase)
    - Spectral sweep (brighter on approach, darker on recede)
    - Micro-texture (sphere grain simulation)
    """

    def __init__(self, model: SpectralModel, seed: int = None):
        self.model = model
        # One evolving generator feeds every layer. Keeping the synthesizer alive
        # across preview requests makes successive chunks part of one stream while
        # an explicit seed still allows reproducible offline comparisons.
        self._rng = np.random.default_rng(seed)
        self._last_wave_phase = None  # filled by _generate_multilayer_envelope
        self._stream_sr = None
        self._stream_sample_index = 0
        self._stream_time = 0.0
        self._fir_states = {}
        self._sos_states = {}
        self._wave_states = {}
        self._swell_states = {}
        self._pan_states = {}
        self._rumble_states = {}
        self._bubble_tails = {}
        self._shore_level = 0.0
        self._master_gain = None

    def _reset_stream_state(self, sr: int) -> None:
        """Reset rate-dependent state when a synthesizer changes sample rate."""
        self._stream_sr = sr
        self._stream_sample_index = 0
        self._stream_time = 0.0
        self._fir_states.clear()
        self._sos_states.clear()
        self._wave_states.clear()
        self._swell_states.clear()
        self._pan_states.clear()
        self._rumble_states.clear()
        self._bubble_tails.clear()
        self._shore_level = 0.0
        self._master_gain = None

    def _filter_stream(self, signal: np.ndarray, coefficients: np.ndarray,
                       state_key: str) -> np.ndarray:
        """FIR-filter one block while retaining its convolution tail."""
        from scipy.signal import lfilter

        state = self._fir_states.get(state_key)
        expected = max(0, len(coefficients) - 1)
        if state is None or len(state) != expected:
            state = np.zeros(expected, dtype=np.float64)
        filtered, state = lfilter(coefficients, [1.0], signal, zi=state)
        self._fir_states[state_key] = state
        return filtered

    def _sos_filter_stream(self, signal: np.ndarray, coefficients: np.ndarray,
                           state_key: str) -> np.ndarray:
        """IIR-filter one block while retaining each section's delay state."""
        from scipy.signal import sosfilt

        state = self._sos_states.get(state_key)
        expected_shape = (len(coefficients), 2)
        if state is None or state.shape != expected_shape:
            state = np.zeros(expected_shape, dtype=np.float64)
        filtered, state = sosfilt(coefficients, signal, zi=state)
        self._sos_states[state_key] = state
        return filtered

    def synthesize(self, duration: float, sr: int = None,
                   stereo: bool = True, intensity: float = 0.5,
                   fast: bool = False) -> np.ndarray:
        """Generate new audio matching the learned model.
        
        fast=True uses single blended FIR + lower sample rate for preview playback.
        """
        if sr is None:
            sr = config.SAMPLE_RATE

        intensity = float(np.clip(intensity, 0.0, 1.0))

        # Fast mode: lower sample rate, single FIR, no micro-texture
        render_sr = 22050 if fast else sr
        n_samples = int(duration * render_sr)
        if self._stream_sr != render_sr:
            self._reset_stream_state(render_sr)
        self._stream_time = self._stream_sample_index / render_sr

        # Build FIR filters
        fir_order = 1023 if fast else 2047
        quiet_fir = self._build_fir(self.model.spectral_flux.get('quiet_spectrum'),
                                    render_sr, fir_order)
        bright_fir = self._build_fir(self.model.spectral_flux.get('bright_spectrum'),
                                     render_sr, fir_order)

        if quiet_fir is None or bright_fir is None:
            avg_fir = self._build_fir(self.model.avg_spectrum, render_sr, fir_order)
            if quiet_fir is None:
                quiet_fir = avg_fir
            if bright_fir is None:
                bright_fir = avg_fir

        if fast:
            # Single blended FIR — avoids 3x convolution
            blend = np.interp(intensity, [0.0, 0.5, 1.0], [0.2, 0.5, 0.8])
            blended_fir = quiet_fir * (1.0 - blend) + bright_fir * blend
            blended_fir = blended_fir / (np.sqrt(np.sum(blended_fir ** 2)) + 1e-10)
            wash_fir = None
        else:
            wash_fir = self._build_wash_fir(render_sr, fir_order)

        if stereo:
            # === SPATIAL OCEAN: far breakers + near lapping + shore wash ===
            # Far layer: main breakers with wide stereo movement
            far_left = self._render_channel(n_samples, render_sr,
                                            quiet_fir if not fast else blended_fir,
                                            bright_fir if not fast else blended_fir,
                                            wash_fir if not fast else blended_fir,
                                            time_offset=0.0,
                                            intensity=intensity, fast=fast,
                                            state_key='far_left')
            far_right = self._render_channel(n_samples, render_sr,
                                             quiet_fir if not fast else blended_fir,
                                             bright_fir if not fast else blended_fir,
                                             wash_fir if not fast else blended_fir,
                                             time_offset=3.5,
                                             intensity=intensity, fast=fast,
                                             state_key='far_right')

            # Spatial panning for far layer — waves sweep L↔R
            far_pan = self._generate_spatial_pan(n_samples, render_sr, intensity, layer='far')
            far_l = far_left * (1.0 - far_pan) + far_right * far_pan * 0.3
            far_r = far_right * far_pan + far_left * (1.0 - far_pan) * 0.3

            # Near layer: small intimate lapping waves (different timing, brighter)
            near_left = self._render_near_layer(n_samples, render_sr,
                                                bright_fir if not fast else blended_fir,
                                                wash_fir if not fast else blended_fir,
                                                intensity=intensity, fast=fast,
                                                state_key='near_left')
            near_right = self._render_near_layer(n_samples, render_sr,
                                                 bright_fir if not fast else blended_fir,
                                                 wash_fir if not fast else blended_fir,
                                                 intensity=intensity, fast=fast,
                                                 state_key='near_right')

            # Near layer panning — narrower, more centered
            near_pan = self._generate_spatial_pan(n_samples, render_sr, intensity, layer='near')
            near_l = near_left * (0.6 + 0.4 * (1.0 - near_pan))
            near_r = near_right * (0.6 + 0.4 * near_pan)

            # Shore wash: high-freq texture from far-layer phase
            shore_l, shore_r = self._generate_shore_wash(n_samples, render_sr, intensity, fast=fast)

            # Mix layers: far (dominant) + near (intimate detail) + shore (texture)
            far_gain = np.interp(intensity, [0.0, 0.5, 1.0], [0.65, 0.70, 0.75])
            near_gain = np.interp(intensity, [0.0, 0.5, 1.0], [0.30, 0.25, 0.20])
            shore_gain = np.interp(intensity, [0.0, 0.5, 1.0], [0.06, 0.08, 0.10])

            left = far_l * far_gain + near_l * near_gain + shore_l * shore_gain
            right = far_r * far_gain + near_r * near_gain + shore_r * shore_gain

            audio = np.stack([left, right], axis=-1)
        else:
            audio = self._render_channel(n_samples, render_sr,
                                         quiet_fir if not fast else blended_fir,
                                         bright_fir if not fast else blended_fir,
                                         wash_fir if not fast else blended_fir,
                                         time_offset=0.0,
                                         intensity=intensity, fast=fast,
                                         state_key='mono')

        audio = self._master_stream_output(audio, intensity)
        self._stream_sample_index += n_samples
        return audio.astype(np.float32)

    def _render_channel(self, n_samples: int, sr: int,
                        quiet_fir: np.ndarray, bright_fir: np.ndarray,
                        wash_fir: np.ndarray,
                        time_offset: float = 0.0,
                        intensity: float = 0.5, fast: bool = False,
                        state_key: str = 'far') -> np.ndarray:
        """Render a single channel of audio."""
        # Draw from the session RNG so equal-length chunks never reuse a texture.
        noise = self._rng.standard_normal(n_samples)

        # Generate multi-layer envelope
        amp_envelope = self._generate_multilayer_envelope(
            n_samples / sr, sr, time_offset=time_offset, intensity=intensity,
            state_key=state_key)

        if fast:
            # Fast mode: single convolution with blended FIR
            audio = self._filter_stream(noise, quiet_fir, f'{state_key}:blend')
            audio = audio * amp_envelope
            # Add rumble layer (works in fast mode too)
            rumble = self._generate_rumble_layer(
                n_samples, sr, amp_envelope, intensity, state_key=state_key)
            audio = audio + rumble
            # Add bubble transients (foam crackle)
            bubbles = self._generate_bubble_transients(
                n_samples, sr, amp_envelope, intensity, state_key=state_key)
            audio = audio + bubbles
        else:
            # Full quality: three-way spectral crossfade driven by per-wave phase
            quiet_stream = self._filter_stream(noise, quiet_fir, f'{state_key}:quiet')
            bright_stream = self._filter_stream(noise, bright_fir, f'{state_key}:bright')
            wash_stream = self._filter_stream(noise, wash_fir, f'{state_key}:wash')

            # Use explicit per-wave spectral phase from envelope generator
            # phase: 0=silence/between waves, rising to 1=peak/break, falling back=foam
            phase = self._last_wave_phase if self._last_wave_phase is not None else np.zeros(n_samples)
            if len(phase) != n_samples:
                phase = np.interp(np.linspace(0, 1, n_samples),
                                  np.linspace(0, 1, len(phase)), phase)

            # Compute phase derivative to distinguish approach vs recede
            phase_deriv = np.gradient(phase, 1.0 / sr)
            phase_deriv_norm = np.tanh(phase_deriv * 0.8)

            # Spectral mixing based on wave lifecycle:
            # Rising phase (approach) → bright spectrum (wave face, broadband)
            # Peak → maximum brightness
            # Falling phase (foam/recede) → wash spectrum (hissy, thin)
            # Low phase (between waves) → quiet spectrum (dark ambient)
            rising = np.clip(phase_deriv_norm, 0, 1)  # positive = approaching
            falling = np.clip(-phase_deriv_norm, 0, 1)  # negative = receding

            bright_mix = phase * rising * 0.8 + phase ** 2 * 0.2  # peaks when phase high & rising
            wash_mix = phase * falling * 0.7  # foam: phase still high but falling
            quiet_mix = np.clip(1.0 - phase, 0.1, 1.0)  # dominant between waves

            quiet_bias = np.interp(intensity, [0.0, 0.5, 1.0], [1.45, 1.0, 0.72])
            bright_bias = np.interp(intensity, [0.0, 0.5, 1.0], [0.55, 1.0, 1.45])
            wash_bias = np.interp(intensity, [0.0, 0.5, 1.0], [0.5, 1.0, 1.35])
            quiet_mix *= quiet_bias
            bright_mix *= bright_bias
            wash_mix *= wash_bias

            total = bright_mix + wash_mix + quiet_mix
            bright_mix /= total
            wash_mix /= total
            quiet_mix /= total

            audio = quiet_stream * quiet_mix + bright_stream * bright_mix + wash_stream * wash_mix
            audio = audio * amp_envelope

            # Add rumble layer
            rumble = self._generate_rumble_layer(
                n_samples, sr, amp_envelope, intensity, state_key=state_key)
            audio = audio + rumble

            # Add bubble transients (foam crackle)
            bubbles = self._generate_bubble_transients(
                n_samples, sr, amp_envelope, intensity, state_key=state_key)
            audio = audio + bubbles

            # Micro-texture (skip in fast mode)
            grain = self._generate_micro_texture(n_samples, sr, state_key=state_key)
            audio = audio * grain

        return audio

    @staticmethod
    def _rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))

    @classmethod
    def _scale_to_rms(cls, signal: np.ndarray, target_rms: float) -> np.ndarray:
        current_rms = cls._rms(signal)
        if current_rms <= 1e-10:
            return signal
        return signal * (target_rms / current_rms)

    @classmethod
    def _master_output(cls, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply stable loudness calibration and a transparent soft ceiling."""
        # Volume is a separate UI control, so intensity primarily changes motion,
        # density and spectrum. A narrow RMS range avoids chunk-to-chunk pumping.
        target_rms = float(np.interp(intensity, [0.0, 1.0], [0.10, 0.12]))
        current_rms = cls._rms(audio)
        if current_rms > 1e-10:
            correction = np.clip(target_rms / current_rms, 0.05, 3.0)
            audio = audio * correction

        # Leave normal samples untouched and round only peaks into a 0.98 ceiling.
        threshold = 0.82
        ceiling = 0.98
        magnitude = np.abs(audio)
        over = magnitude > threshold
        if np.any(over):
            span = ceiling - threshold
            limited = threshold + span * np.tanh((magnitude[over] - threshold) / span)
            audio = audio.copy()
            audio[over] = np.sign(audio[over]) * limited
        return audio

    def _master_stream_output(self, audio: np.ndarray,
                              intensity: float) -> np.ndarray:
        """Master a block without introducing a gain step at its boundary."""
        target_rms = float(np.interp(intensity, [0.0, 1.0], [0.10, 0.12]))
        current_rms = self._rms(audio)
        desired_gain = 1.0
        if current_rms > 1e-10:
            desired_gain = float(np.clip(target_rms / current_rms, 0.05, 3.0))

        start_gain = desired_gain if self._master_gain is None else self._master_gain
        ramp_samples = min(len(audio), max(1, int((self._stream_sr or 44100) * 3.0)))
        gains = np.full(len(audio), desired_gain, dtype=np.float64)
        if ramp_samples > 1:
            gains[:ramp_samples] = np.linspace(
                start_gain, desired_gain, ramp_samples, endpoint=True)
        if audio.ndim > 1:
            audio = audio * gains[:, np.newaxis]
        else:
            audio = audio * gains
        self._master_gain = desired_gain

        threshold = 0.82
        ceiling = 0.98
        magnitude = np.abs(audio)
        over = magnitude > threshold
        if np.any(over):
            span = ceiling - threshold
            limited = threshold + span * np.tanh((magnitude[over] - threshold) / span)
            audio = audio.copy()
            audio[over] = np.sign(audio[over]) * limited
        return audio

    def _model_period_scale(self) -> float:
        """Map the learned amplitude period onto safe synthesis timing bounds."""
        learned_period = float(self.model.amp_stats.get('period_seconds', 2.5))
        if not np.isfinite(learned_period):
            learned_period = 2.5
        # Values around one second are commonly the detector's lower-bound result;
        # keep them usable without turning the drum into rapid-fire noise.
        learned_period = float(np.clip(learned_period, 1.5, 6.0))
        return float(np.clip(learned_period / 2.5, 0.65, 2.4))

    def _build_wash_fir(self, sr: int, order: int) -> np.ndarray:
        """
        Build a FIR for the 'receding wash' — thinner, higher character.
        Takes the bright spectrum and boosts 2-6kHz while cutting lows,
        simulating spheres rolling back (lighter, hissier).
        """
        spectrum = self.model.spectral_flux.get('bright_spectrum')
        if spectrum is None:
            spectrum = self.model.avg_spectrum

        n_freq = order // 2 + 1
        learned_freqs = self.model.freqs
        target_freqs = np.linspace(0, sr / 2, n_freq)

        if len(learned_freqs) != n_freq:
            interp_func = interp1d(learned_freqs, spectrum,
                                   kind='linear', fill_value=0, bounds_error=False)
            mag_response = interp_func(target_freqs)
        else:
            mag_response = spectrum.copy()

        mag_response = mag_response / (np.max(mag_response) + 1e-10)

        # Wash character: cut below 800 Hz, boost 2-6 kHz
        wash_shape = np.ones_like(target_freqs)
        # Roll off lows
        low_mask = target_freqs < 800
        wash_shape[low_mask] = (target_freqs[low_mask] / 800.0) ** 1.5
        # Gentle boost in 2-6 kHz
        mid_mask = (target_freqs >= 2000) & (target_freqs <= 6000)
        wash_shape[mid_mask] *= 1.4

        mag_response = mag_response * wash_shape

        # Apply same perceptual tilt
        ref_freq = 1000.0
        tilt = np.ones_like(target_freqs)
        above_ref = target_freqs > ref_freq
        tilt[above_ref] = (ref_freq / target_freqs[above_ref]) ** 0.5
        mag_response = mag_response * tilt

        # Build FIR
        impulse = np.fft.irfft(mag_response, n=order)
        fir = np.roll(impulse, order // 2)
        window = np.hanning(order)
        fir = fir * window
        fir = fir / (np.sqrt(np.sum(fir ** 2)) + 1e-10)

        return fir

    def _generate_spatial_pan(self, n_samples: int, sr: int,
                              intensity: float, layer: str = 'far') -> np.ndarray:
        """
        Generate a smooth stereo panning signal for spatial wave movement.
        
        Far layer: wide sweeps (0.1-0.9 range) at wave-group rate
        Near layer: narrower, faster random movement (0.3-0.7 range)
        
        Returns array in [0,1] where 0.5=center, 0=full left, 1=full right.
        """
        start = self._stream_time
        times = start + np.arange(n_samples, dtype=np.float64) / sr
        if layer == 'far':
            value_range = (0.15, 0.85)
            duration_range = (6.0, 12.0)
        else:
            value_range = (0.3, 0.7)
            duration_range = (0.8, 2.2)

        state = self._pan_states.get(layer)
        if state is None:
            initial = float(self._rng.uniform(*value_range))
            state = {
                'start': start,
                'end': start + float(self._rng.uniform(*duration_range)),
                'from': initial,
                'to': float(self._rng.uniform(*value_range)),
            }
            self._pan_states[layer] = state

        pan = np.empty(n_samples, dtype=np.float64)
        filled = np.zeros(n_samples, dtype=bool)
        while not np.all(filled):
            mask = (~filled) & (times >= state['start']) & (times < state['end'])
            if np.any(mask):
                progress = (times[mask] - state['start']) / (
                    state['end'] - state['start'])
                eased = 0.5 - 0.5 * np.cos(np.pi * progress)
                pan[mask] = state['from'] + (state['to'] - state['from']) * eased
                filled[mask] = True
            if np.all(filled):
                break
            previous_target = state['to']
            state.update({
                'start': state['end'],
                'end': state['end'] + float(self._rng.uniform(*duration_range)),
                'from': previous_target,
                'to': float(self._rng.uniform(*value_range)),
            })

        return pan

    def _render_near_layer(self, n_samples: int, sr: int,
                           bright_fir: np.ndarray, wash_fir: np.ndarray,
                           intensity: float = 0.5, fast: bool = False,
                           state_key: str = 'near') -> np.ndarray:
        """
        Render the 'near' layer — small intimate waves lapping at your feet.
        
        Characteristics vs far layer:
        - Shorter wave periods (1.5-4s vs 3-8s)
        - Less dynamic range (quieter peaks, higher floor — always audible)
        - Brighter spectrum (more high-frequency sand/water detail)
        - Independent timing from far layer
        """
        noise = self._rng.standard_normal(n_samples)

        # Generate near-wave envelope with faster, smaller waves
        near_env = self._generate_near_envelope(
            n_samples / sr, sr, intensity, state_key=state_key)

        if fast:
            audio = self._filter_stream(noise, bright_fir, f'{state_key}:blend')
            audio = audio * near_env
        else:
            # Near waves are mostly bright + wash (little quiet component)
            bright_stream = self._filter_stream(
                noise, bright_fir, f'{state_key}:bright')
            if wash_fir is not None:
                wash_stream = self._filter_stream(
                    noise, wash_fir, f'{state_key}:wash')
                # Near waves: 60% bright + 40% wash (hissy sand sound)
                audio = bright_stream * 0.6 + wash_stream * 0.4
            else:
                audio = bright_stream
            audio = audio * near_env

        return audio

    def _generate_near_envelope(self, duration: float, sr: int,
                                intensity: float,
                                state_key: str = 'near') -> np.ndarray:
        """
        Envelope for near lapping waves — faster, smaller, more regular.
        Period: 1.5-4s, amplitude: 0.3-0.7 (never as loud as far breakers).
        """
        start = self._stream_time
        end = start + duration
        key = f'near:{state_key}'
        state = self._wave_states.get(key)
        if state is None:
            state = {
                'next_time': start + float(self._rng.uniform(0.3, 1.5)),
                'events': [],
            }
            self._wave_states[key] = state

        period_scale = np.sqrt(self._model_period_scale())
        min_period = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [3.0, 2.2, 1.5])) * period_scale
        max_period = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [5.0, 4.0, 3.0])) * period_scale

        while state['next_time'] < end:
            period = float(self._rng.uniform(min_period, max_period))
            active_duration = period * 0.7
            event = {
                'start': state['next_time'],
                'duration': active_duration,
                'rise': float(self._rng.uniform(0.35, 0.50)),
                'amplitude': float(self._rng.uniform(0.3, 0.7)),
            }
            state['events'].append(event)
            state['next_time'] += active_duration + float(self._rng.uniform(0.2, 0.8))

        state['events'] = [
            event for event in state['events']
            if event['start'] + event['duration'] >= start
        ]

        ctrl_rate = 20.0
        ctrl_times = start + np.arange(
            max(2, int(np.ceil(duration * ctrl_rate)) + 1)) / ctrl_rate
        envelope = np.zeros(len(ctrl_times), dtype=np.float64)
        for event in state['events']:
            progress = (ctrl_times - event['start']) / event['duration']
            active = (progress >= 0.0) & (progress <= 1.0)
            if not np.any(active):
                continue
            local = progress[active]
            rise = event['rise']
            shape = np.where(
                local < rise,
                np.power(np.clip(local / rise, 0.0, 1.0), 1.2),
                np.power(np.clip((1.0 - local) / (1.0 - rise), 0.0, 1.0), 0.9),
            )
            envelope[active] = np.maximum(
                envelope[active], shape * event['amplitude'])

        floor = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.10, 0.15, 0.20]))
        envelope = np.maximum(envelope, floor)
        audio_times = start + np.arange(int(duration * sr), dtype=np.float64) / sr
        return np.interp(audio_times, ctrl_times, envelope)

    def _generate_shore_wash(self, n_samples: int, sr: int,
                             intensity: float,
                             fast: bool = False) -> tuple:
        """
        Shore wash layer — the distinctive 'shhhh' of water rushing up sand
        and the gurgling drain-back.
        
        Triggered by the far-layer wave phase (self._last_wave_phase).
        High-frequency (1-8kHz) with long foam tails.
        Returns (left, right) tuple.
        """
        from scipy.signal import butter

        nyq = sr / 2

        # Use the wave phase from far layer to time shore interaction
        phase = self._last_wave_phase if self._last_wave_phase is not None else np.zeros(n_samples)
        if len(phase) != n_samples:
            phase = np.interp(np.linspace(0, 1, n_samples),
                              np.linspace(0, 1, len(phase)), phase)

        # Shore wash happens during and after wave peak (phase > 0.4)
        # with a long extended tail (foam persists on sand)
        # Work at control rate (100 Hz) for performance, then upsample
        ctrl_rate = 100
        n_ctrl = max(4, int(n_samples / sr * ctrl_rate))
        phase_ctrl = np.interp(np.linspace(0, 1, n_ctrl),
                               np.linspace(0, 1, n_samples), phase)

        shore_ctrl = np.zeros(n_ctrl)
        decay_rate = np.interp(intensity, [0.0, 0.5, 1.0], [0.3, 0.5, 0.8])
        decay_per_step = decay_rate / ctrl_rate
        rise_alpha = 1.0 - 0.95 ** (sr / ctrl_rate)  # adapt smoothing to ctrl rate

        running_level = self._shore_level
        for i in range(n_ctrl):
            input_energy = max(0, phase_ctrl[i] - 0.3) * 2.0
            if input_energy > running_level:
                running_level = running_level * (1.0 - rise_alpha) + input_energy * rise_alpha
            else:
                running_level = max(0, running_level - decay_per_step)
            shore_ctrl[i] = running_level
        self._shore_level = running_level

        # Upsample to audio rate
        shore_trigger = np.interp(np.linspace(0, 1, n_samples),
                                  np.linspace(0, 1, n_ctrl), shore_ctrl)

        shore_trigger = np.clip(shore_trigger, 0.0, 1.0)

        # Generate shore noise — band-limited 800Hz-8kHz (sand/foam character)
        noise_l = self._rng.standard_normal(n_samples)
        noise_r = self._rng.standard_normal(n_samples)

        # Bandpass for shore character
        low_cut = min(800.0 / nyq, 0.95)
        high_cut = min(8000.0 / nyq, 0.99)
        if low_cut < high_cut:
            sos = butter(3, [low_cut, high_cut], btype='band', output='sos')
            shore_noise_l = self._sos_filter_stream(
                noise_l, sos, 'shore:left')
            shore_noise_r = self._sos_filter_stream(
                noise_r, sos, 'shore:right')
        else:
            shore_noise_l = noise_l
            shore_noise_r = noise_r

        # Apply shore trigger envelope
        shore_l = shore_noise_l * shore_trigger
        shore_r = shore_noise_r * shore_trigger

        shore_level = np.interp(intensity, [0.0, 0.5, 1.0], [0.18, 0.24, 0.30])
        shore_l *= shore_level
        shore_r *= shore_level

        return shore_l, shore_r

    def _generate_micro_texture(self, n_samples: int, sr: int,
                                state_key: str = 'texture') -> np.ndarray:
        """Subtle 5-20 Hz amplitude flicker simulating sphere contacts."""
        from scipy.signal import butter

        mod_noise = self._rng.standard_normal(n_samples)
        nyq = sr / 2
        sos = butter(2, [5.0 / nyq, 20.0 / nyq], btype='band', output='sos')
        grain_signal = self._sos_filter_stream(
            mod_noise, sos, f'{state_key}:micro')

        return 1.0 + 0.10 * np.tanh(grain_signal * 15.0)

    def _generate_rumble_layer(self, n_samples: int, sr: int,
                                amp_envelope: np.ndarray,
                                intensity: float = 0.5,
                                state_key: str = 'rumble') -> np.ndarray:
        """
        Sub-bass rumble layer (20-80 Hz) with two components:
        1. Independent slow undertow — its own swell cycle (8-20s period)
        2. Peak-coupled rumble — louder when main waves crash (envelope peaks)
        
        Returns audio to be mixed into the main signal.
        """
        from scipy.signal import butter, lfilter

        nyq = sr / 2
        if nyq <= 80:
            return np.zeros(n_samples)

        # Generate bass noise source
        noise = self._rng.standard_normal(n_samples)

        # Bandpass 20-80 Hz
        low = max(20.0 / nyq, 0.001)
        high = min(80.0 / nyq, 0.99)
        sos = butter(3, [low, high], btype='band', output='sos')
        bass_noise = self._sos_filter_stream(
            noise, sos, f'{state_key}:rumble-filter')

        # --- Component 1: Independent undertow swell ---
        # Slow envelope with its own random period (8-20s)
        state = self._rumble_states.get(state_key)
        if state is None:
            state = {
                'period': float(self._rng.uniform(8.0, 20.0)),
                'phase': float(self._rng.uniform(0, 2 * np.pi)),
                'gain': None,
            }
            self._rumble_states[state_key] = state
        times = self._stream_time + np.arange(n_samples, dtype=np.float64) / sr
        undertow_env_full = 0.5 + 0.5 * np.sin(
            2 * np.pi * times / state['period'] + state['phase'])

        # --- Component 2: Peak-coupled rumble ---
        # Follow the main envelope but emphasize peaks
        # Square the envelope to emphasize louder moments
        peak_env = amp_envelope ** 2
        # Smooth it slightly for a lagging bass response
        smoothing_seconds = 0.3
        alpha = 1.0 - np.exp(-1.0 / (sr * smoothing_seconds))
        peak_key = f'{state_key}:rumble-envelope'
        peak_state = self._fir_states.get(peak_key)
        if peak_state is None or len(peak_state) != 1:
            peak_state = np.zeros(1, dtype=np.float64)
        peak_env, peak_state = lfilter(
            [alpha], [1.0, -(1.0 - alpha)], peak_env, zi=peak_state)
        self._fir_states[peak_key] = peak_state

        # Combine: 60% independent undertow + 40% peak-coupled
        combined_env = 0.6 * undertow_env_full + 0.4 * peak_env
        combined_env = np.clip(combined_env, 0.0, 1.0)
        rumble = bass_noise * combined_env

        # Calibrate energy with a slow gain transition instead of a block reset.
        rumble_rms = np.interp(intensity, [0.0, 0.5, 1.0], [0.006, 0.014, 0.025])
        current_rms = self._rms(rumble)
        desired_gain = 1.0 if current_rms <= 1e-10 else float(
            np.clip(rumble_rms / current_rms, 0.05, 20.0))
        start_gain = desired_gain if state['gain'] is None else state['gain']
        gain_ramp = np.linspace(start_gain, desired_gain, n_samples, endpoint=True)
        rumble = rumble * gain_ramp
        state['gain'] = desired_gain

        return rumble

    def _generate_bubble_transients(self, n_samples: int, sr: int,
                                     amp_envelope: np.ndarray,
                                     intensity: float = 0.5,
                                     state_key: str = 'bubbles') -> np.ndarray:
        """
        Sparse bubble transients based on Minnaert resonance physics.
        
        Breaking waves entrain air bubbles that ring as damped harmonic
        oscillators. Frequency is determined by bubble radius:
            f = 3.26 / radius  (Hz·m at standard conditions)
        
        Bubble radii in surf: 0.5mm–10mm → frequencies 326 Hz–6.5 kHz.
        Each bubble = damped sinusoid. Density follows the amplitude envelope
        (more bubbles during wave crashes, fewer during calm).
        """
        rng = self._rng
        output = np.zeros(n_samples)
        previous_tail = self._bubble_tails.pop(state_key, None)
        if previous_tail is not None:
            tail_length = min(n_samples, len(previous_tail))
            output[:tail_length] += previous_tail[:tail_length]
            if len(previous_tail) > n_samples:
                self._bubble_tails[state_key] = previous_tail[n_samples:]

        # Bubble event density (events per second) scales with intensity
        # Calm: sparse crackle; stormy: dense foam fizz
        base_density = np.interp(intensity, [0.0, 0.5, 1.0], [3.0, 15.0, 60.0])

        # Downsample envelope to control rate for scheduling efficiency
        ctrl_rate = 20  # Hz
        n_ctrl = max(1, int(n_samples / sr * ctrl_rate))
        ctrl_indices = np.linspace(0, n_samples - 1, n_ctrl).astype(int)
        ctrl_env = amp_envelope[ctrl_indices]
        # Normalize to [0, 1]
        ctrl_env = ctrl_env / (np.max(ctrl_env) + 1e-10)

        # Schedule bubble events — probability proportional to envelope²
        # (bubbles happen during breaking, not during silence)
        ctrl_duration = n_samples / sr / n_ctrl  # seconds per control point
        
        for i in range(n_ctrl):
            # Local density modulated by envelope (squared for emphasis on peaks)
            local_density = base_density * (ctrl_env[i] ** 2)
            n_events = rng.poisson(local_density * ctrl_duration)
            
            if n_events == 0:
                continue

            # Time window for this control segment
            t_start_sample = ctrl_indices[i]
            t_end_sample = ctrl_indices[min(i + 1, n_ctrl - 1)] if i < n_ctrl - 1 else n_samples

            for _ in range(n_events):
                # Random bubble radius: log-uniform between 0.5mm and 10mm
                # Smaller bubbles more likely (power-law distribution)
                radius = 10 ** rng.uniform(-3.3, -2.0)  # 0.5mm to 10mm
                
                # Minnaert frequency
                freq = 3.26 / radius  # Hz
                if freq > sr / 2 - 100:  # respect Nyquist
                    continue

                # Damping: smaller bubbles damp faster
                # Typical damping ratio 0.05–0.25
                damping = rng.uniform(0.08, 0.22)
                
                # Duration of the transient (until amplitude < -40dB)
                # e^(-ζωt) = 0.01 → t = ln(100) / (ζω)
                omega = 2 * np.pi * freq
                bubble_duration = min(4.6 / (damping * omega), 0.15)  # cap at 150ms
                bubble_samples = int(bubble_duration * sr)
                
                if bubble_samples < 4:
                    continue

                # Random onset within this control segment
                onset = rng.integers(t_start_sample, max(t_start_sample + 1, t_end_sample))
                # Synthesize damped sinusoid
                t = np.arange(bubble_samples) / sr
                bubble = np.exp(-damping * omega * t) * np.sin(omega * t)
                
                # Random amplitude (smaller bubbles tend to be quieter)
                amp = rng.uniform(0.3, 1.0) * (radius / 0.01) ** 0.3
                
                available = min(bubble_samples, n_samples - onset)
                output[onset:onset + available] += bubble[:available] * amp
                if available < bubble_samples:
                    tail = bubble[available:] * amp
                    existing = self._bubble_tails.get(state_key)
                    if existing is None:
                        self._bubble_tails[state_key] = tail.copy()
                    else:
                        if len(existing) < len(tail):
                            existing = np.pad(existing, (0, len(tail) - len(existing)))
                        existing[:len(tail)] += tail
                        self._bubble_tails[state_key] = existing

        # Fixed gain preserves natural event-density variation between chunks.
        bubble_gain = np.interp(intensity, [0.0, 0.5, 1.0], [0.003, 0.008, 0.016])
        output *= bubble_gain

        return output

    def _build_fir(self, spectrum: np.ndarray, sr: int, order: int) -> np.ndarray:
        """Build a linear-phase FIR filter from a learned magnitude spectrum."""
        if spectrum is None:
            return None

        n_freq = order // 2 + 1
        learned_freqs = self.model.freqs
        target_freqs = np.linspace(0, sr / 2, n_freq)

        if len(learned_freqs) != n_freq:
            interp_func = interp1d(learned_freqs, spectrum,
                                   kind='linear', fill_value=0, bounds_error=False)
            mag_response = interp_func(target_freqs)
        else:
            mag_response = spectrum.copy()

        mag_response = mag_response / (np.max(mag_response) + 1e-10)

        # Perceptual correction: pink-noise tilt above 1kHz
        ref_freq = 1000.0
        tilt = np.ones_like(target_freqs)
        above_ref = target_freqs > ref_freq
        tilt[above_ref] = (ref_freq / target_freqs[above_ref]) ** 0.5
        mag_response = mag_response * tilt

        impulse = np.fft.irfft(mag_response, n=order)
        fir = np.roll(impulse, order // 2)
        window = np.hanning(order)
        fir = fir * window
        fir = fir / (np.sqrt(np.sum(fir ** 2)) + 1e-10)

        return fir

    def _generate_multilayer_envelope(self, duration: float, sr: int,
                                       time_offset: float = 0.0,
                                       intensity: float = 0.5,
                                       state_key: str = 'far') -> np.ndarray:
        """
        Multi-layer envelope with oceanographic wave statistics:
        - Layer 1 (swell): Slow undulation — wave SETS (groups)
        - Layer 2 (waves): Individual waves with Rayleigh-distributed heights,
          grouped into sets of 3-7 where middle waves are largest.
        - Per-wave asymmetric shape: fast rise (approach/break) + slow decay (foam/recede)
        
        Returns amplitude envelope at audio sample rate.
        Also stores self._wave_events for spectral evolution tracking.
        """
        stats = self.model.amp_stats
        intensity = float(np.clip(intensity, 0.0, 1.0))
        start = self._stream_time
        end = start + duration
        period_scale = self._model_period_scale()

        wave_key = f'far:{state_key}'
        state = self._wave_states.get(wave_key)
        if state is None:
            state = {
                'next_time': start + float(self._rng.uniform(0.2, 1.0)),
                'events': [],
                'group_remaining': 0,
                'group_position': 0,
                'group_total': 0,
            }
            self._wave_states[wave_key] = state

        min_period = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [4.0, 3.0, 2.0])) * period_scale
        max_period = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [8.0, 6.0, 4.5])) * period_scale
        learned_variation = stats.get('std', 0.2) / max(
            stats.get('mean', 0.4), 0.05)
        variation_scale = float(np.clip(learned_variation / 0.5, 0.75, 1.25))
        rayleigh_scale = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.4, 0.7, 0.95])) * variation_scale
        inter_group_pause = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [4.0, 2.5, 1.0]))

        while state['next_time'] < end:
            if state['group_remaining'] <= 0:
                state['group_total'] = int(self._rng.integers(3, 8))
                state['group_remaining'] = state['group_total']
                state['group_position'] = 0

            raw_height = float(self._rng.rayleigh(rayleigh_scale))
            amplitude = float(np.clip(raw_height, 0.15, 1.4))
            center = (state['group_total'] - 1) / 2.0
            group_factor = 1.0 + 0.3 * np.exp(
                -((state['group_position'] - center) ** 2)
                / (max(1.0, state['group_total'] / 3.0) ** 2))
            amplitude *= group_factor
            base_period = float(self._rng.uniform(min_period, max_period))
            active_duration = base_period * (0.8 + 0.4 * (amplitude / 1.4))
            rise = float(self._rng.uniform(
                np.interp(intensity, [0.0, 1.0], [0.30, 0.20]),
                np.interp(intensity, [0.0, 1.0], [0.42, 0.35]),
            ))
            state['events'].append({
                'start': state['next_time'],
                'duration': active_duration,
                'rise': rise,
                'amplitude': amplitude,
            })

            state['group_remaining'] -= 1
            state['group_position'] += 1
            if state['group_remaining'] <= 0:
                pause = float(self._rng.uniform(
                    inter_group_pause * 0.7, inter_group_pause * 1.5))
            else:
                base_pause = float(np.interp(
                    intensity, [0.0, 0.5, 1.0], [1.0, 0.5, 0.15]))
                pause = float(self._rng.uniform(
                    base_pause * 0.6, base_pause * 1.4))
            state['next_time'] += active_duration + pause

        state['events'] = [
            event for event in state['events']
            if event['start'] + event['duration'] >= start
        ]

        ctrl_rate = 20.0
        ctrl_times = start + np.arange(
            max(2, int(np.ceil(duration * ctrl_rate)) + 1)) / ctrl_rate
        waves = np.zeros(len(ctrl_times), dtype=np.float64)
        wave_phase = np.zeros(len(ctrl_times), dtype=np.float64)
        for event in state['events']:
            progress = (ctrl_times - event['start']) / event['duration']
            active = (progress >= 0.0) & (progress <= 1.0)
            if not np.any(active):
                continue
            local = progress[active]
            rise = event['rise']
            shape = np.where(
                local < rise,
                np.power(np.clip(local / rise, 0.0, 1.0), 1.5),
                np.power(np.clip((1.0 - local) / (1.0 - rise), 0.0, 1.0), 0.7),
            )
            phase = np.where(
                local < rise,
                np.clip(local / rise, 0.0, 1.0),
                np.clip((1.0 - local) / (1.0 - rise), 0.0, 1.0),
            )
            waves[active] += shape * event['amplitude']
            wave_phase[active] = np.maximum(wave_phase[active], phase)
        waves = np.clip(waves / 1.82, 0.0, 1.0)

        swell_key = f'far:{state_key}'
        swell_layers = self._swell_states.get(swell_key)
        if swell_layers is None:
            layer_count = int(round(np.interp(
                intensity, [0.0, 0.5, 1.0], [2.0, 3.0, 4.0])))
            period_scale_root = np.sqrt(period_scale)
            minimum = float(np.interp(
                intensity, [0.0, 0.5, 1.0], [10.0, 8.0, 6.0])) * period_scale_root
            maximum = float(np.interp(
                intensity, [0.0, 0.5, 1.0], [18.0, 15.0, 11.0])) * period_scale_root
            amplitude_max = float(np.interp(
                intensity, [0.0, 0.5, 1.0], [0.45, 0.6, 0.85]))
            swell_layers = [
                {
                    'period': float(self._rng.uniform(minimum, maximum)),
                    'phase': float(self._rng.uniform(0, 2 * np.pi)),
                    'amplitude': float(self._rng.uniform(0.22, amplitude_max)),
                }
                for _ in range(layer_count)
            ]
            self._swell_states[swell_key] = swell_layers

        swell = np.zeros(len(ctrl_times), dtype=np.float64)
        swell_weight = 0.0
        for layer in swell_layers:
            swell += layer['amplitude'] * (
                0.5 + 0.5 * np.sin(
                    2 * np.pi * (ctrl_times + time_offset) / layer['period']
                    + layer['phase']))
            swell_weight += layer['amplitude']
        swell /= swell_weight + 1e-10
        swell_floor = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.16, 0.3, 0.42]))
        swell_depth = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.42, 0.7, 0.9]))
        swell = swell_floor + swell_depth * swell

        envelope = waves * swell
        amp_min = stats['min'] * float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.06, 0.15, 0.22]))
        amp_max = min(stats['max'] * float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.72, 1.1, 1.35])), 1.0)
        envelope = amp_min + envelope * (amp_max - amp_min)
        ambient_floor = float(np.interp(
            intensity, [0.0, 0.5, 1.0], [0.08, 0.12, 0.18]))
        envelope = np.maximum(envelope, ambient_floor)

        audio_times = start + np.arange(int(duration * sr), dtype=np.float64) / sr
        self._last_wave_phase = np.interp(audio_times, ctrl_times, wave_phase)
        return np.interp(audio_times, ctrl_times, envelope)


def learn_from_file(wav_path: str) -> SpectralModel:
    """Load a WAV file and learn a SpectralModel from it."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample to our sample rate if different
    if sr != config.SAMPLE_RATE:
        from scipy.signal import resample
        n_target = int(len(audio) * config.SAMPLE_RATE / sr)
        audio = resample(audio, n_target)
        sr = config.SAMPLE_RATE

    return SpectralModel.from_audio(audio, sr)
