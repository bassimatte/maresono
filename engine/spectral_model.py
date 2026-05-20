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

    def __init__(self, model: SpectralModel):
        self.model = model

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
            mono = self._render_channel(n_samples, render_sr,
                                        quiet_fir if not fast else blended_fir,
                                        bright_fir if not fast else blended_fir,
                                        wash_fir if not fast else blended_fir,
                                        seed_offset=0, time_offset=0.0,
                                        intensity=intensity, fast=fast)
            side = self._render_channel(n_samples, render_sr,
                                        quiet_fir if not fast else blended_fir,
                                        bright_fir if not fast else blended_fir,
                                        wash_fir if not fast else blended_fir,
                                        seed_offset=77, time_offset=12.0,
                                        intensity=intensity, fast=fast)
            width = 0.05
            left = mono * (1.0 - width) + side * width
            right = mono * (1.0 - width) + side * (-width)
            peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
            if peak > 0:
                target_peak = np.interp(intensity, [0.0, 0.5, 1.0], [0.55, 0.9, 0.98])
                left = left / peak * target_peak
                right = right / peak * target_peak
            audio = np.stack([left, right], axis=-1)
        else:
            audio = self._render_channel(n_samples, render_sr,
                                         quiet_fir if not fast else blended_fir,
                                         bright_fir if not fast else blended_fir,
                                         wash_fir if not fast else blended_fir,
                                         seed_offset=0, time_offset=0.0,
                                         intensity=intensity, fast=fast)

        return audio.astype(np.float32)

    def _render_channel(self, n_samples: int, sr: int,
                        quiet_fir: np.ndarray, bright_fir: np.ndarray,
                        wash_fir: np.ndarray,
                        seed_offset: int = 0, time_offset: float = 0.0,
                        intensity: float = 0.5, fast: bool = False) -> np.ndarray:
        """Render a single channel of audio."""
        from scipy.signal import fftconvolve

        # Different noise per channel
        rng = np.random.default_rng(seed=None if seed_offset == 0 else seed_offset + int(n_samples))
        noise = rng.standard_normal(n_samples)

        # Generate multi-layer envelope
        amp_envelope = self._generate_multilayer_envelope(
            n_samples / sr, sr, time_offset=time_offset, intensity=intensity)

        if fast:
            # Fast mode: single convolution with blended FIR
            audio = fftconvolve(noise, quiet_fir, mode='same')
            audio = audio * amp_envelope
            # Add rumble layer (works in fast mode too)
            rumble = self._generate_rumble_layer(n_samples, sr, amp_envelope, intensity)
            audio = audio + rumble
        else:
            # Full quality: three-way spectral crossfade
            quiet_stream = fftconvolve(noise, quiet_fir, mode='same')
            bright_stream = fftconvolve(noise, bright_fir, mode='same')
            wash_stream = fftconvolve(noise, wash_fir, mode='same')

            env_deriv = np.gradient(amp_envelope, 1.0 / sr)
            d_max = np.max(np.abs(env_deriv)) + 1e-10
            phase = 0.5 + 0.5 * (env_deriv / d_max)

            env_norm = (amp_envelope - np.min(amp_envelope)) / (np.max(amp_envelope) - np.min(amp_envelope) + 1e-10)

            rising = np.clip(phase - 0.5, 0, 0.5) * 2.0
            falling = np.clip(0.5 - phase, 0, 0.5) * 2.0

            bright_mix = rising * 0.7 + env_norm * 0.3
            wash_mix = falling * 0.6
            quiet_mix = np.clip(1.0 - bright_mix - wash_mix, 0.1, 1.0)

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
            rumble = self._generate_rumble_layer(n_samples, sr, amp_envelope, intensity)
            audio = audio + rumble

            # Micro-texture (skip in fast mode)
            grain = self._generate_micro_texture(n_samples, sr)
            audio = audio * grain

        # Normalize while preserving the intended intensity range
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = np.interp(intensity, [0.0, 0.5, 1.0], [0.55, 0.9, 0.98])
            audio = audio / peak * target_peak

        return audio

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

    def _generate_micro_texture(self, n_samples: int, sr: int) -> np.ndarray:
        """Subtle 5-20 Hz amplitude flicker simulating sphere contacts."""
        from scipy.signal import butter, sosfilt

        mod_noise = np.random.randn(n_samples)
        nyq = sr / 2
        sos = butter(2, [5.0 / nyq, 20.0 / nyq], btype='band', output='sos')
        grain_signal = sosfilt(sos, mod_noise)
        grain_signal = grain_signal / (np.max(np.abs(grain_signal)) + 1e-10)

        return 1.0 + 0.10 * grain_signal

    def _generate_rumble_layer(self, n_samples: int, sr: int,
                                amp_envelope: np.ndarray,
                                intensity: float = 0.5) -> np.ndarray:
        """
        Sub-bass rumble layer (20-80 Hz) with two components:
        1. Independent slow undertow — its own swell cycle (8-20s period)
        2. Peak-coupled rumble — louder when main waves crash (envelope peaks)
        
        Returns audio to be mixed into the main signal.
        """
        from scipy.signal import butter, sosfilt

        nyq = sr / 2
        if nyq <= 80:
            return np.zeros(n_samples)

        # Generate bass noise source
        noise = np.random.randn(n_samples)

        # Bandpass 20-80 Hz
        low = max(20.0 / nyq, 0.001)
        high = min(80.0 / nyq, 0.99)
        sos = butter(3, [low, high], btype='band', output='sos')
        bass_noise = sosfilt(sos, noise)

        # --- Component 1: Independent undertow swell ---
        # Slow envelope with its own random period (8-20s)
        control_rate = 10  # Hz
        n_ctrl = max(4, int(n_samples / sr * control_rate))
        period = np.random.uniform(8.0, 20.0)
        t_ctrl = np.linspace(0, n_samples / sr, n_ctrl)
        # Random phase so each chunk is different
        phase = np.random.uniform(0, 2 * np.pi)
        undertow_env = 0.5 + 0.5 * np.sin(2 * np.pi * t_ctrl / period + phase)
        # Add some randomness
        undertow_env *= (0.7 + 0.3 * np.random.rand(n_ctrl))
        # Interpolate to sample rate
        t_samples = np.linspace(0, n_samples / sr, n_samples)
        undertow_env_full = np.interp(t_samples, t_ctrl, undertow_env)

        # --- Component 2: Peak-coupled rumble ---
        # Follow the main envelope but emphasize peaks
        # Square the envelope to emphasize louder moments
        peak_env = amp_envelope ** 2
        # Smooth it slightly for a lagging bass response
        from scipy.ndimage import uniform_filter1d
        smooth_len = min(int(sr * 0.3), n_samples)  # 300ms smoothing
        if smooth_len > 1:
            peak_env = uniform_filter1d(peak_env, smooth_len)

        # Combine: 60% independent undertow + 40% peak-coupled
        combined_env = 0.6 * undertow_env_full + 0.4 * peak_env
        combined_env = combined_env / (np.max(combined_env) + 1e-10)

        # Apply envelope to bass noise
        rumble = bass_noise * combined_env

        # Scale by intensity — more intensity = more rumble
        # At intensity 0: very subtle (0.04), at 1.0: prominent (0.18)
        rumble_gain = np.interp(intensity, [0.0, 0.5, 1.0], [0.04, 0.10, 0.18])
        rumble = rumble / (np.max(np.abs(rumble)) + 1e-10) * rumble_gain

        return rumble

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
                                       intensity: float = 0.5) -> np.ndarray:
        """
        Multi-layer envelope:
        - Layer 1 (swell): Slow 8-15s undulation — wave SETS (groups of waves)
        - Layer 2 (waves): Individual wave events (2-8s) riding on the swell
        
        The swell modulates the amplitude of individual waves, so waves
        are bigger during swell peaks and smaller during swell troughs.
        """
        stats = self.model.amp_stats
        intensity = float(np.clip(intensity, 0.0, 1.0))
        n_points = int(duration * 10)  # 10 Hz control rate
        dt = 1.0 / 10.0  # seconds per point

        # --- Layer 1: Slow swell (8-15s period) ---
        t = np.linspace(0, duration, n_points, endpoint=False)
        swell = np.zeros(n_points)
        n_swell_layers = int(round(np.interp(intensity, [0.0, 0.5, 1.0], [2.0, 3.0, 4.0])))
        swell_min_period = np.interp(intensity, [0.0, 0.5, 1.0], [10.0, 8.0, 6.0])
        swell_max_period = np.interp(intensity, [0.0, 0.5, 1.0], [18.0, 15.0, 11.0])
        swell_floor = np.interp(intensity, [0.0, 0.5, 1.0], [0.16, 0.3, 0.42])
        swell_depth = np.interp(intensity, [0.0, 0.5, 1.0], [0.42, 0.7, 0.9])
        for _ in range(n_swell_layers):
            swell_period = np.random.uniform(swell_min_period, swell_max_period)
            swell_phase = np.random.uniform(0, 2 * np.pi) + time_offset * 2 * np.pi / swell_period
            swell_amp = np.random.uniform(0.22, np.interp(intensity, [0.0, 0.5, 1.0], [0.45, 0.6, 0.85]))
            swell += swell_amp * (0.5 + 0.5 * np.sin(2 * np.pi * t / swell_period + swell_phase))

        swell = swell / (np.max(swell) + 1e-10)
        swell = swell_floor + swell_depth * swell

        # --- Layer 2: Individual wave events ---
        waves = np.zeros(n_points)
        current_time = np.random.uniform(0.3, np.interp(intensity, [0.0, 1.0], [1.3, 0.45])) + time_offset
        min_wave_duration = np.interp(intensity, [0.0, 0.5, 1.0], [1.5, 2.5, 4.0])
        max_wave_duration = np.interp(intensity, [0.0, 0.5, 1.0], [3.0, 7.0, 8.0])
        min_wave_amp = np.interp(intensity, [0.0, 0.5, 1.0], [0.28, 0.5, 0.72])
        max_wave_amp = np.interp(intensity, [0.0, 0.5, 1.0], [0.62, 1.0, 1.28])
        long_pause_chance = np.interp(intensity, [0.0, 0.5, 1.0], [0.55, 0.25, 0.1])

        while current_time < duration - 1.0:
            active_dur = np.random.uniform(min_wave_duration, max_wave_duration)
            wave_amp = np.random.uniform(min_wave_amp, max_wave_amp)
            rise_frac = np.random.uniform(0.18, np.interp(intensity, [0.0, 1.0], [0.34, 0.5]))

            i_start = int(current_time / dt)
            i_end = int((current_time + active_dur) / dt)
            i_start = max(0, min(i_start, n_points - 1))
            i_end = max(i_start + 1, min(i_end, n_points))

            n_wave = i_end - i_start
            if n_wave > 2:
                rise_len = max(2, int(n_wave * rise_frac))
                decay_len = max(1, n_wave - rise_len)

                rise = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, rise_len)))
                decay = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, decay_len)))

                wave_shape = np.concatenate([rise, decay]) * wave_amp
                end_idx = min(i_start + len(wave_shape), n_points)
                waves[i_start:end_idx] += wave_shape[:end_idx - i_start]

            if np.random.random() < long_pause_chance:
                pause = np.random.uniform(
                    np.interp(intensity, [0.0, 0.5, 1.0], [3.0, 2.0, 1.2]),
                    np.interp(intensity, [0.0, 0.5, 1.0], [5.0, 3.5, 2.0]),
                )
            else:
                pause = np.random.uniform(
                    np.interp(intensity, [0.0, 0.5, 1.0], [1.2, 0.6, 0.2]),
                    np.interp(intensity, [0.0, 0.5, 1.0], [2.8, 1.8, 0.8]),
                )

            current_time += active_dur + pause

        # Normalize waves to [0, 1]
        peak_val = np.max(waves)
        if peak_val > 0:
            waves = waves / peak_val

        # --- Combine: waves modulated by swell ---
        envelope = waves * swell

        # Scale to learned dynamic range — never fully silent
        amp_min = stats['min'] * np.interp(intensity, [0.0, 0.5, 1.0], [0.06, 0.15, 0.22])
        amp_max = min(stats['max'] * np.interp(intensity, [0.0, 0.5, 1.0], [0.72, 1.1, 1.35]), 1.0)
        envelope = amp_min + envelope * (amp_max - amp_min)

        # Add ambient floor so it never goes fully silent
        ambient_floor = np.interp(intensity, [0.0, 0.5, 1.0], [0.08, 0.12, 0.18])
        envelope = np.maximum(envelope, ambient_floor)

        # Smooth
        from scipy.ndimage import uniform_filter1d
        envelope = uniform_filter1d(envelope, size=8)

        # Upsample to audio rate
        n_audio = int(duration * sr)
        envelope_interp = interp1d(np.linspace(0, 1, len(envelope)),
                                   envelope, kind='cubic')
        return envelope_interp(np.linspace(0, 1, n_audio))


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
