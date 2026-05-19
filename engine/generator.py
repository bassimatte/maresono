"""
engine/generator.py
-------------------
Procedural preset generator for Maresono.
Creates randomized ocean presets with mood biases.
"""

import random
import time
from pathlib import Path
from typing import Optional

import yaml


MOODS = {
    "calm": {
        "wave_period": (8, 14),
        "risacca_mix": (0.3, 0.5),
        "foam_mix": (0.1, 0.2),
        "reverb_mix": (0.2, 0.4),
        "binaural_hz": (4, 6),
        "description": "Gentle Mediterranean calm",
    },
    "storm": {
        "wave_period": (3, 6),
        "risacca_mix": (0.8, 1.0),
        "foam_mix": (0.5, 0.8),
        "reverb_mix": (0.1, 0.2),
        "binaural_hz": (7, 10),
        "description": "Powerful Atlantic storm",
    },
    "meditation": {
        "wave_period": (10, 18),
        "risacca_mix": (0.2, 0.4),
        "foam_mix": (0.05, 0.15),
        "reverb_mix": (0.3, 0.5),
        "binaural_hz": (4, 5),
        "description": "Deep theta meditation waves",
    },
    "cove": {
        "wave_period": (6, 10),
        "risacca_mix": (0.5, 0.7),
        "foam_mix": (0.2, 0.4),
        "reverb_mix": (0.4, 0.6),
        "binaural_hz": (5, 7),
        "description": "Sheltered cove with echoing walls",
    },
    "crystal": {
        "wave_period": (8, 12),
        "risacca_mix": (0.3, 0.5),
        "foam_mix": (0.1, 0.2),
        "reverb_mix": (0.2, 0.35),
        "binaural_hz": (5, 7),
        "description": "Crystal-infused healing waves (Quarzo Rosa)",
    },
}

_ADJECTIVES = [
    "Eternal", "Ancient", "Gentle", "Whispering", "Golden", "Silver",
    "Midnight", "Emerald", "Crystal", "Forgotten", "Sacred", "Lunar",
    "Solar", "Infinite", "Primordial", "Amber", "Obsidian", "Ethereal",
]

_NOUNS = [
    "Shore", "Tide", "Surf", "Lagoon", "Reef", "Current",
    "Bay", "Cove", "Coast", "Horizon", "Depths", "Passage",
    "Grotto", "Sanctuary", "Threshold", "Dream", "Embrace", "Breath",
]


def get_available_moods() -> list:
    return list(MOODS.keys())


def _random_name() -> str:
    return f"{random.choice(_ADJECTIVES)} {random.choice(_NOUNS)}"


def generate_preset(mood: Optional[str] = None, seed: Optional[int] = None) -> dict:
    """Generate a random ocean preset."""
    if seed is not None:
        random.seed(seed)

    if mood is None:
        mood = random.choice(list(MOODS.keys()))

    m = MOODS[mood]
    name = _random_name()
    slug = name.lower().replace(" ", "_")

    wave_period = random.uniform(*m["wave_period"])
    risacca_mix = random.uniform(*m["risacca_mix"])
    foam_mix = random.uniform(*m["foam_mix"])
    reverb_mix = random.uniform(*m["reverb_mix"])
    binaural_hz = random.uniform(*m["binaural_hz"])

    preset = {
        "meta": {
            "name": name,
            "slug": slug,
            "category": mood,
            "mood": [mood],
            "description": m["description"],
            "author": "Maresono",
        },
        "duration": 60,
        "layers": [
            {
                "name": "Ocean Swell",
                "type": "wave",
                "enabled": True,
                "mix": 1.0,
                "pan": random.uniform(-0.3, 0.0),
                "noise": {
                    "white": round(random.uniform(0.02, 0.1), 3),
                    "pink": round(random.uniform(0.25, 0.5), 3),
                    "brown": round(random.uniform(0.4, 0.7), 3),
                },
                "frequency": {
                    "low": random.randint(40, 100),
                    "high": random.randint(800, 1500),
                    "order": random.choice([2, 3, 4]),
                },
                "envelope": {
                    "cycle_period": round(wave_period, 1),
                    "attack_ratio": round(random.uniform(0.35, 0.55), 2),
                    "randomize": round(random.uniform(0.2, 0.4), 2),
                },
            },
            {
                "name": "Risacca",
                "type": "risacca",
                "enabled": True,
                "mix": round(risacca_mix, 2),
                "pan": random.uniform(0.0, 0.4),
                "noise": {
                    "white": round(random.uniform(0.1, 0.3), 3),
                    "pink": round(random.uniform(0.4, 0.6), 3),
                    "brown": round(random.uniform(0.1, 0.4), 3),
                },
                "frequency": {
                    "low": random.randint(300, 800),
                    "high": random.randint(4000, 8000),
                    "order": random.choice([2, 3]),
                },
                "envelope": {
                    "cycle_period": round(wave_period * random.uniform(0.6, 0.9), 1),
                    "delay": round(random.uniform(1.0, 3.5), 1),
                },
            },
            {
                "name": "Foam",
                "type": "foam",
                "enabled": foam_mix > 0.1,
                "mix": round(foam_mix, 2),
                "pan": round(random.uniform(-0.2, 0.2), 2),
                "frequency": {
                    "low": random.randint(2000, 4000),
                    "high": random.randint(10000, 14000),
                },
                "envelope": {
                    "cycle_period": round(wave_period * random.uniform(0.8, 1.1), 1),
                },
            },
        ],
        "binaural": {
            "enabled": mood in ("meditation", "crystal", "calm"),
            "method": "both",
            "beat_hz": round(binaural_hz, 1),
            "carrier_hz": random.choice([100, 120, 150, 180, 200]),
            "carrier_amplitude": round(random.uniform(0.04, 0.08), 3),
            "mod_depth": round(random.uniform(0.1, 0.2), 2),
        },
        "reverb": {
            "enabled": True,
            "space": random.choice(["beach", "cove", "cliff"]) if mood != "cove" else "cove",
            "mix": round(reverb_mix, 2),
            "decay_trim": round(random.uniform(0.8, 1.2), 1),
        },
    }

    # Add crystal layer for crystal mood
    if mood == "crystal":
        preset["layers"].append({
            "name": "Crystal Resonance",
            "type": "crystal",
            "enabled": True,
            "mix": round(random.uniform(0.15, 0.3), 2),
            "pan": 0.0,
            "frequency": {"fundamental": random.choice([432, 528, 639, 741])},
            "harmonics": random.randint(4, 7),
            "harmonic_decay": round(random.uniform(0.5, 0.7), 2),
            "envelope": {"cycle_period": round(random.uniform(10, 16), 1)},
        })

    return preset


def save_generated_preset(preset: dict, directory: Path) -> Path:
    """Save a generated preset to YAML."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    slug = preset["meta"]["slug"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.yaml"
    path = directory / filename

    with path.open("w", encoding="utf-8") as f:
        yaml.dump(preset, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return path
