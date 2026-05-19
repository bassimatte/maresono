from __future__ import annotations

import asyncio
import io
import os
import re
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import soundfile as sf

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as exc:
    raise ImportError(
        "Web UI requires fastapi and uvicorn.\n"
        "Install with: pip install fastapi uvicorn[standard]"
    ) from exc

from . import config
from .spectral_model import SpectralModel, SpectralSynthesizer

_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _ENGINE_DIR.parent
_STATIC_DIR = _ENGINE_DIR / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"
_EXPORTS_DIR = _REPO_DIR / "exports"
_DEFAULT_MODELS_DIR = _REPO_DIR / "models"
_PREVIEW_SECONDS = 20.0

_EXPORTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Maresono", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RenderRequest(BaseModel):
    model: str
    duration: float = Field(default=60.0, gt=0)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    preview_duration: float | None = Field(default=None, gt=0)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "maresono"


def _models_dir() -> Path:
    override = os.getenv("MARESONO_MODELS_DIR")
    return Path(override).expanduser() if override else _DEFAULT_MODELS_DIR


def _available_model_paths() -> list[Path]:
    models_dir = _models_dir()
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.npz"), key=lambda path: path.name.lower())


def _resolve_model_path(filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name.endswith('.npz'):
        safe_name = Path(safe_name).stem + '.npz'
    models_root = _models_dir().resolve()
    path = (models_root / safe_name).resolve()
    try:
        path.relative_to(models_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid model path.") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {safe_name}")
    return path


@lru_cache(maxsize=16)
def _load_learned_model_from_path(path_str: str):
    return SpectralModel.load(Path(path_str))


def _load_learned_model(filename: str):
    path = _resolve_model_path(filename)
    return _load_learned_model_from_path(str(path))


def _render_audio(filename: str, duration: float, intensity: float, fast: bool = False):
    model = _load_learned_model(filename)
    synthesizer = SpectralSynthesizer(model)
    sr = 22050 if fast else config.SAMPLE_RATE
    audio = synthesizer.synthesize(
        duration=duration,
        sr=sr,
        stereo=True,
        intensity=intensity,
        fast=fast,
    )
    return audio, sr


def _audio_to_wav_buffer(audio, sr: int = None) -> io.BytesIO:
    if sr is None:
        sr = config.SAMPLE_RATE
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype=config.BIT_DEPTH)
    buffer.seek(0)
    return buffer


def _export_filename(filename: str, duration: float, intensity: float) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = _slugify(Path(filename).stem)
    minutes = duration / 60.0
    duration_tag = f"{minutes:g}m" if minutes >= 1 else f"{duration:g}s"
    intensity_tag = f"i{int(round(intensity * 100)):02d}"
    return f"{stem}-{duration_tag}-{intensity_tag}-{stamp}.wav"


def _render_package(request: RenderRequest, *, preview: bool) -> tuple[io.BytesIO, str]:
    duration = _PREVIEW_SECONDS if preview else request.duration
    # Allow client to request shorter preview (for quick-start first chunk)
    if preview and hasattr(request, 'preview_duration') and request.preview_duration:
        duration = min(request.preview_duration, _PREVIEW_SECONDS)
    # Preview uses fast mode (22050 Hz, single FIR) for speed
    audio, sr = _render_audio(request.model, duration, request.intensity, fast=preview)
    filename = _export_filename(request.model, duration, request.intensity)

    if not preview:
        export_path = _EXPORTS_DIR / filename
        sf.write(str(export_path), audio, sr, format="WAV", subtype=config.BIT_DEPTH)

    return _audio_to_wav_buffer(audio, sr), filename


@app.get("/")
def index():
    if not _INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Static UI not found.")
    return FileResponse(_INDEX_FILE)


@app.get("/api/models")
def list_models():
    return [path.name for path in _available_model_paths()]


@app.post("/api/render")
async def render_audio(request: RenderRequest):
    buffer, filename = await asyncio.to_thread(_render_package, request, preview=False)
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/preview")
async def preview_audio(request: RenderRequest):
    buffer, filename = await asyncio.to_thread(_render_package, request, preview=True)
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="preview-{filename}"'},
    )


def run_server(host: str = "127.0.0.1", port: int | None = None) -> None:
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_port = int(os.getenv("PORT", port or 8000))
    uvicorn.run(app, host=host, port=resolved_port, log_level="info")


def launch_gui(host: str = "127.0.0.1", port: int | None = None) -> None:
    import webbrowser

    resolved_port = int(os.getenv("PORT", port or 8000))
    print("Maresono — Web UI")
    print(f"   http://{host}:{resolved_port}")
    print("   Press Ctrl+C to stop.\n")

    if host != "0.0.0.0":
        def _open_browser() -> None:
            import time

            time.sleep(1.2)
            webbrowser.open(f"http://{host}:{resolved_port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    run_server(host=host, port=resolved_port)


if __name__ == "__main__":
    run_server()
