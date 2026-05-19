"""
engine/spheres.py
-----------------
Particle-based sphere simulation for Maresono.

Models two layers of spheres rolling on membranes inside an ocean drum:
  - Large spheres (glass/plastic): lighter, move FAST, produce soft/low sound
  - Small spheres (metallic): heavier for their size, move SLOWER, brilliant sound

Physical construction:
  - Frame: 4 layers — birch (outer), poplar (inner, where spheres touch)
    The soft poplar produces gentle thuds when spheres reach the rim edge.
  - 3 membranes separate 2 chambers:
    - Membrane 1 (top): thin, high resonance — transparent, bright
    - Membrane 2 (middle): shared between chambers — warm, damped
    - Membrane 3 (bottom): thin, high resonance — transparent, bright
  - Chamber 1 (between mem 1-2): large plastic/glass spheres → wave
  - Chamber 2 (between mem 2-3): small metallic spheres → risacca

Sphere behavior:
  Large spheres move fast, produce soft sound.
  Small metallic spheres move slower, produce brilliant sound.

Physics model:
  The drum is circular (~35cm). When tilted, spheres orbit along the rim.
  Sound swells as sphere cluster passes the listening point.
"""

import numpy as np
from . import config
from .filters import Filters


class MembraneModel:
    """
    Models the resonant character of the drum's 3 membranes.
    
    Each membrane has a different resonant frequency and damping:
    - Membrane 1 & 3 (outer): thin, transparent → higher resonance, less damping
    - Membrane 2 (middle, shared): thicker → lower resonance, more damping
    
    The rolling sound is colored by the membrane the spheres are rolling ON:
    - Large spheres roll on membrane 2 (bottom of chamber 1) → warmer
    - Small spheres roll on membrane 2 (top of chamber 2) → same membrane!
    
    But the RADIATED sound also passes through the outer membrane, adding
    its resonant coloring.
    """
    
    # Membrane resonant peaks (Hz) and Q factors
    MEMBRANE_OUTER = {'peaks': [800, 2200, 4500], 'q': 3.0}   # thin, transparent
    MEMBRANE_MIDDLE = {'peaks': [400, 1200, 2800], 'q': 2.0}  # thicker, warmer
    
    @staticmethod
    def apply(audio: np.ndarray, chamber: str = 'large') -> np.ndarray:
        """
        Apply membrane resonance coloring to sphere audio.
        
        chamber='large': rolling on middle membrane, radiating through outer
        chamber='small': rolling on middle membrane, radiating through outer
        """
        from scipy.signal import iirpeak, sosfilt
        
        # The rolling surface is always membrane 2 (middle)
        # But radiation goes through the nearest outer membrane
        rolling_membrane = MembraneModel.MEMBRANE_MIDDLE
        radiation_membrane = MembraneModel.MEMBRANE_OUTER
        
        result = audio.copy()
        
        # Apply subtle resonant peaks from the rolling surface
        for peak_hz in rolling_membrane['peaks']:
            if peak_hz < config.SAMPLE_RATE * 0.45:
                bw = peak_hz / rolling_membrane['q']
                b, a = iirpeak(peak_hz, rolling_membrane['q'], config.SAMPLE_RATE)
                # Convert to mild boost (mix 20% resonance)
                from scipy.signal import lfilter
                resonant = lfilter(b, a, result)
                result = result * 0.85 + resonant * 0.15
        
        # Apply radiation membrane coloring (subtler — just 10%)
        for peak_hz in radiation_membrane['peaks']:
            if peak_hz < config.SAMPLE_RATE * 0.45:
                b, a = iirpeak(peak_hz, radiation_membrane['q'], config.SAMPLE_RATE)
                from scipy.signal import lfilter
                resonant = lfilter(b, a, result)
                result = result * 0.92 + resonant * 0.08
        
        return result


class FrameContact:
    """
    Models the soft poplar frame contact sound.
    
    When spheres reach the outer edge of their orbit (touching the 4cm wide
    poplar frame), they produce a soft, muted thud — NOT a harsh click.
    Poplar is a soft wood that absorbs impact energy.
    
    The sound is: short burst of low-frequency noise (< 500 Hz),
    with fast attack and exponential decay (~30ms).
    """
    
    @staticmethod
    def generate(ang_position: np.ndarray, speed: np.ndarray, 
                 sphere_mass: float, n_samples: int) -> np.ndarray:
        """
        Generate frame contact sounds based on sphere position and speed.
        
        Contacts happen when angular velocity reverses direction (sphere 
        reaches the extent of its arc and "bumps" the frame gently).
        """
        output = np.zeros(n_samples)
        
        # Detect velocity reversals (direction changes = frame contact points)
        # Use zero-crossings of angular velocity derivative (acceleration sign change)
        vel_sign = np.sign(speed)
        # Actually: detect when speed drops sharply (sphere decelerating at frame)
        speed_diff = np.diff(speed, prepend=speed[0])
        
        # Find sharp deceleration moments (potential frame contacts)
        threshold = -np.std(speed_diff) * 1.5
        contacts = np.where(speed_diff < threshold)[0]
        
        # Limit to reasonable number of contacts (not every sample)
        min_gap = int(0.3 * config.SAMPLE_RATE)  # min 300ms between contacts
        filtered_contacts = []
        last = -min_gap
        for c in contacts:
            if c - last >= min_gap:
                filtered_contacts.append(c)
                last = c
        
        if not filtered_contacts:
            return output
        
        # Generate each contact sound
        contact_duration = int(0.04 * config.SAMPLE_RATE)  # 40ms
        
        for contact_sample in filtered_contacts:
            # Contact intensity proportional to speed at impact
            impact_speed = float(speed[contact_sample]) if contact_sample < len(speed) else 0
            intensity = np.clip(impact_speed * sphere_mass * 2.0, 0, 1.0)
            
            if intensity < 0.05:
                continue
            
            # Soft poplar thud: low-pass filtered noise with fast decay
            end = min(contact_sample + contact_duration, n_samples)
            length = end - contact_sample
            
            # Short noise burst
            burst = np.random.randn(length)
            
            # Exponential decay envelope (fast — 20ms decay)
            decay = np.exp(-np.arange(length) / (0.015 * config.SAMPLE_RATE))
            burst *= decay * intensity * 0.3  # 0.3 = subtle mix level
            
            # Low-pass at ~400 Hz (poplar absorbs highs)
            if length > 10:
                burst = Filters.lowpass(burst, 400, order=2)
            
            output[contact_sample:end] += burst
        
        return output


class LargeSphereLayer:
    """
    Simulates large glass/plastic spheres rolling inside a circular drum (~35cm).
    
    These move FAST and produce a SOFT sound.
    They're plastic/glass — lighter than metallic — faster response to tilt.
    The sound is a low, soft whoosh — the "wave" component.
    
    Circular orbit model: spheres don't hit boundaries, they roll along the rim.
    Sound swells as the cluster passes the listening point.
    """

    def __init__(self, cfg: dict, duration: float):
        self.duration = duration
        self.n_samples = int(duration * config.SAMPLE_RATE)
        self.n_spheres = int(cfg.get("count", 12))
        self.membrane_resonance = float(cfg.get("membrane_resonance", 600))
        self.mass = float(cfg.get("mass", 1.5))  # plastic/glass — lighter!
        self.diameter = float(cfg.get("diameter", 0.35))

    def build(self, tilt_envelope: np.ndarray) -> np.ndarray:
        """
        Generate audio from large spheres orbiting inside circular drum.
        
        The tilt_envelope drives angular velocity — spheres circulate along
        the rim. Sound intensity depends on proximity to a "listening point"
        (top of the membrane).
        """
        from scipy.signal import lfilter

        output = np.zeros(self.n_samples)
        dt = 1.0 / config.SAMPLE_RATE
        radius = self.diameter / 2.0

        # Pre-roll: repeat first 2s of tilt to warm up IIR filters
        preroll_samples = min(int(2.0 * config.SAMPLE_RATE), self.n_samples)
        preroll = tilt_envelope[:preroll_samples]
        extended_tilt = np.concatenate([preroll, tilt_envelope])

        # DAMPED oscillator — NOT continuous rotation.
        # Friction = 0.995 means velocity decays quickly, tracking the tilt.
        # Spheres swing back and forth like a pendulum, with natural pauses
        # at the extremes (like a real wave: build → peak → recede → pause).
        angular_friction = 0.995

        for i in range(self.n_spheres):
            sphere_resonance = self.membrane_resonance + np.random.uniform(-25, 25)
            amplitude = np.random.uniform(0.6, 1.0)

            # Angular force from tilt (divided by mass and radius)
            angular_force = extended_tilt * 9.8 / (self.mass * radius) * dt

            # Angular velocity — damped, so it tracks tilt direction (not accumulates)
            ang_velocity_full = lfilter([1.0], [1.0, -angular_friction], angular_force)
            ang_velocity = ang_velocity_full[preroll_samples:]

            # Position: also damped — oscillates, doesn't wrap around
            # This gives natural back-and-forth arc (not full rotation)
            position_friction = 0.999
            ang_position = lfilter([1.0], [1.0, -position_friction], ang_velocity * dt)

            # Sound swells when sphere is moving (speed), and fades when it pauses
            # No "proximity" trick needed — the natural speed variation IS the wave
            speed = np.abs(ang_velocity) * radius
            speed_smooth = self._smooth(speed, int(0.05 * config.SAMPLE_RATE))

            # Stereo position from angular position (subtle panning)
            # But main volume driver is SPEED (not proximity)

            # Rolling friction sound — filtered noise modulated by speed
            noise = np.random.randn(self.n_samples)

            chunk_size = int(0.05 * config.SAMPLE_RATE)
            sphere_audio = np.zeros(self.n_samples)

            for c in range(0, self.n_samples, chunk_size):
                end = min(c + chunk_size, self.n_samples)
                avg_speed = float(np.mean(speed_smooth[c:end]))
                # Bandpass: real drum energy is 250-2500 Hz
                center = sphere_resonance + avg_speed * 600
                center = np.clip(center, 250, 3000)
                bw = center * 0.8
                low = max(center - bw / 2, 150)
                high = min(center + bw / 2, config.SAMPLE_RATE * 0.45)
                sphere_audio[c:end] = Filters.bandpass(noise[c:end], low, high, order=2)

            # Volume follows speed — natural wave envelope
            sphere_audio *= speed_smooth * amplitude

            # Frame contact: soft poplar thud when sphere decelerates at rim
            frame_sound = FrameContact.generate(
                ang_position, speed, self.mass, self.n_samples)
            sphere_audio += frame_sound * amplitude

            output += sphere_audio

        # Apply membrane resonance coloring
        output = MembraneModel.apply(output, chamber='large')

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0:
            output /= peak

        return output

    @staticmethod
    def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')


class SmallSphereLayer:
    """
    Simulates small metallic spheres rolling inside the circular drum.
    
    These move SLOWER and produce a BRILLIANT sound.
    Metal — heavier per size — more inertia — slower response to tilt.
    Metal-on-membrane friction — brighter, more resonant sound.
    
    Same circular orbit physics, but higher friction coefficient and
    higher membrane resonance give the "risacca/crest" shimmer.
    """

    def __init__(self, cfg: dict, duration: float):
        self.duration = duration
        self.n_samples = int(duration * config.SAMPLE_RATE)
        self.n_spheres = int(cfg.get("count", 20))
        self.brightness = float(cfg.get("brightness", 0.7))
        self.scatter = float(cfg.get("scatter", 0.4))
        self.membrane_resonance = float(cfg.get("membrane_resonance", 2500))
        self.mass = float(cfg.get("mass", 0.8))  # metallic — heavier per size!
        self.diameter = float(cfg.get("diameter", 0.35))

    def build(self, tilt_envelope: np.ndarray) -> np.ndarray:
        """
        Generate audio from small spheres orbiting inside the drum.
        Same tilt drives them, but lighter mass = faster response.
        Scatter means each sphere has a slightly different orbit.
        """
        from scipy.signal import lfilter

        output = np.zeros(self.n_samples)
        dt = 1.0 / config.SAMPLE_RATE
        radius = self.diameter / 2.0

        # Pre-roll for IIR warm-up
        preroll_samples = min(int(2.0 * config.SAMPLE_RATE), self.n_samples)

        # Damped oscillation — metallic spheres are heavier so even more damped
        angular_friction = 0.993

        for i in range(self.n_spheres):
            sphere_resonance = self.membrane_resonance + np.random.uniform(-500, 500)
            amplitude = np.random.uniform(0.4, 1.0)

            # Each small sphere has individual drift added to the shared tilt
            individual_drift = self._random_drift() * self.scatter
            effective_tilt = tilt_envelope * (1.0 - self.scatter * 0.5) + individual_drift

            # Pre-roll
            extended_tilt = np.concatenate([effective_tilt[:preroll_samples], effective_tilt])

            # Angular force (heavier mass → weaker response → slower)
            angular_force = extended_tilt * 9.8 / (self.mass * radius) * dt

            # Damped angular velocity — oscillates with tilt, doesn't accumulate
            ang_velocity_full = lfilter([1.0], [1.0, -angular_friction], angular_force)
            ang_velocity = ang_velocity_full[preroll_samples:]

            # Damped position — back and forth arc
            position_friction = 0.998
            ang_position = lfilter([1.0], [1.0, -position_friction], ang_velocity * dt)

            # Speed drives volume — natural wave-like envelope
            speed = np.abs(ang_velocity) * radius
            speed_smooth = self._smooth(speed, int(0.03 * config.SAMPLE_RATE))

            # Rolling friction sound — bandpass around membrane resonance
            noise = np.random.randn(self.n_samples)

            chunk_size = int(0.04 * config.SAMPLE_RATE)
            sphere_audio = np.zeros(self.n_samples)

            for c in range(0, self.n_samples, chunk_size):
                end = min(c + chunk_size, self.n_samples)
                avg_speed = float(np.mean(speed_smooth[c:end]))
                center = sphere_resonance + avg_speed * 800
                center = np.clip(center, 800, 8000)
                bw = center * 0.5
                low = max(center - bw / 2, 200)
                high = min(center + bw / 2, config.SAMPLE_RATE * 0.45)
                sphere_audio[c:end] = Filters.bandpass(noise[c:end], low, high, order=2)

            # Volume follows speed — natural swell and fade
            sphere_audio *= speed_smooth * amplitude

            # Frame contact: metallic spheres make slightly brighter thuds on poplar
            frame_sound = FrameContact.generate(
                ang_position, speed, self.mass, self.n_samples)
            sphere_audio += frame_sound * amplitude

            output += sphere_audio

        # Apply membrane resonance coloring
        output = MembraneModel.apply(output, chamber='small')

        peak = np.max(np.abs(output))
        if peak > 0:
            output /= peak

        return output

    def _random_drift(self) -> np.ndarray:
        """Smooth random drift for individual sphere variation."""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        drift = np.zeros(self.n_samples)
        for _ in range(3):
            freq = np.random.uniform(0.05, 0.3)
            phase = np.random.uniform(0, 2 * np.pi)
            drift += np.sin(2 * np.pi * freq * t + phase)
        drift /= np.max(np.abs(drift)) + 1e-10
        return drift

    @staticmethod
    def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')


def generate_tilt_envelope(
    duration: float,
    cycle_period: float = 8.0,
    randomize: float = 0.3,
) -> np.ndarray:
    """
    Generate the drum tilt envelope — simulates how a player slowly tilts
    the ocean drum to create wave motion.
    
    The motion is slow and deliberate: tilt one way... hold... tilt back.
    A full cycle (tilt left → tilt right → back) takes cycle_period seconds.
    
    Returns values in [-1, 1] where:
      -1 = fully tilted left
      +1 = fully tilted right
       0 = level
    """
    n_samples = int(duration * config.SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Primary slow tilt motion (fundamental)
    base_freq = 1.0 / cycle_period
    tilt = np.sin(2 * np.pi * base_freq * t)
    
    # Add subtle sub-harmonics for organic feel (player isn't a metronome)
    tilt += 0.15 * np.sin(2 * np.pi * base_freq * 0.5 * t + np.random.uniform(0, np.pi))
    tilt += 0.08 * np.sin(2 * np.pi * base_freq * 1.7 * t + np.random.uniform(0, np.pi))
    
    # Smooth out any harsh transitions (player moves gently)
    from scipy.ndimage import uniform_filter1d
    smooth_window = int(0.3 * config.SAMPLE_RATE)  # 300ms smoothing
    tilt = uniform_filter1d(tilt, smooth_window)
    
    # Normalize to [-1, 1]
    tilt /= np.max(np.abs(tilt)) + 1e-10
    
    # Apply gentle randomization to amplitude (some tilts bigger than others)
    if randomize > 0:
        n_cycles = int(duration / cycle_period) + 1
        amp_variation = np.interp(
            np.linspace(0, n_cycles, n_samples),
            np.arange(n_cycles),
            1.0 - randomize + 2 * randomize * np.random.random(n_cycles),
        )
        tilt *= amp_variation
    
    return tilt
