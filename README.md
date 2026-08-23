# Maresono

*Mare + Suono — The Sound of the Sea*

A digital ocean wave sound synthesizer that learns from real ocean drum recordings
and generates infinite, non-looping wave audio using spectral resynthesis.

## Features

- **Spectral learning**: Learns the timbre from real recordings (any WAV file)
- **FIR-based synthesis**: Smooth, artifact-free sound (no frame boundaries)
- **Multi-layer envelope**: Slow swells + individual waves for natural rhythm
- **Spectral sweep**: Waves get brighter on approach, darker on recede
- **Receding wash**: Different spectral character for wave decay phase
- **Stereo field**: Subtle spatial movement
- **Micro-texture**: Simulates individual sphere grain contacts
- **Continuous synthesis state**: Wave timing, filter tails, undertow, shore wash,
  and stereo movement persist across web playback chunks
- **FastAPI web GUI** with animated breathing circle
- **Multiple export formats** (WAV, FLAC, OGG)
- **Real-time preview** via sounddevice

## Usage

```bash
python main.py                              # process all presets
python main.py --list                       # list available presets
python main.py --preset presets/calm.yaml   # single preset
python main.py --name "Storm"              # find preset by name
python main.py --duration 300              # 5 minutes
python main.py --format flac              # export as FLAC
python main.py --preview --name "Calm"    # real-time to speakers
python main.py --gui                      # launch web UI in browser
```

## Install

```bash
pip install -r requirements.txt
```

## Pre-computed Models

Maresono now loads pre-computed spectral models from `models/*.npz` at runtime.
The original WAV reference recordings are only needed during the training step.

To add or refresh models:

```bash
python train.py
python train.py --input ../reference_recordings --output models
```

Workflow:
1. Place a new reference WAV in `../reference_recordings/`
2. Run `python train.py`
3. Commit the generated `.npz` file from `models/`

At runtime, Maresono only needs the small `.npz` model files; the WAV files are not required.
The `MARESONO_MODELS_DIR` environment variable can still override the default model directory.

## How It Works

Maresono uses an analysis-resynthesis approach:
1. Learns the spectral envelope (frequency shape) from a reference recording
2. Pre-computes and saves the learned spectral statistics as a compact `.npz` model
3. Loads the `.npz` model at runtime
4. Generates new audio by shaping noise with FIR filters matched to the learned spectrum
5. Applies multi-layer amplitude modulation for natural wave dynamics
6. Crossfades between quiet/bright/wash spectra based on wave phase
