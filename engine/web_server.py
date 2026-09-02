from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import io
import os
import re
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import soundfile as sf

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response, StreamingResponse
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
_FAVICON_FILE = _STATIC_DIR / "favicon.svg"
_EXPORTS_DIR = _REPO_DIR / "exports"
_DEFAULT_MODELS_DIR = _REPO_DIR / "models"
_PREVIEW_SECONDS = 30.0
_PREVIEW_SESSION_TTL = 15 * 60.0
_PREVIEW_CLEANUP_INTERVAL = 60.0
_MAX_PREVIEW_SESSIONS = 8

_EXPORTS_DIR.mkdir(exist_ok=True)

class RenderRequest(BaseModel):
    model: str
    duration: float = Field(default=60.0, gt=0)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    preview_duration: Optional[float] = Field(default=None, gt=0)
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    chunk_index: Optional[int] = Field(default=None, ge=0)


class _PreviewSession:
    def __init__(self, model_name: str, intensity: float, model: SpectralModel):
        self.model_name = model_name
        self.intensity = intensity
        self.synthesizer = SpectralSynthesizer(model)
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.last_chunk_index = -1
        self.cached_key = None
        self.cached_wav = None
        self.cached_filename = None


_preview_sessions: dict[str, _PreviewSession] = {}
_preview_sessions_lock = threading.Lock()


def _cleanup_expired_preview_sessions(now: float | None = None) -> int:
    if now is None:
        now = time.monotonic()
    with _preview_sessions_lock:
        expired = [
            key for key, session in _preview_sessions.items()
            if now - session.last_used > _PREVIEW_SESSION_TTL
        ]
        for key in expired:
            _preview_sessions.pop(key, None)
    return len(expired)


def _close_preview_session(session_id: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        raise HTTPException(status_code=400, detail="Invalid preview session ID.")
    with _preview_sessions_lock:
        return _preview_sessions.pop(session_id, None) is not None


async def _preview_session_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(_PREVIEW_CLEANUP_INTERVAL)
        _cleanup_expired_preview_sessions()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    cleanup_task = asyncio.create_task(_preview_session_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        with _preview_sessions_lock:
            _preview_sessions.clear()


app = FastAPI(title="Maresono", version="2.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _render_audio(filename: str, duration: float, intensity: float, fast: bool = False,
                  synthesizer: SpectralSynthesizer = None):
    if synthesizer is None:
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


def _get_preview_session(request: RenderRequest) -> _PreviewSession:
    session_id = request.session_id or ''
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        raise HTTPException(status_code=400, detail="Invalid preview session ID.")

    now = time.monotonic()
    _cleanup_expired_preview_sessions(now)
    with _preview_sessions_lock:
        session = _preview_sessions.get(session_id)
        if session is not None:
            same_model = session.model_name == request.model
            same_intensity = abs(session.intensity - request.intensity) <= 1e-6
            if not same_model or not same_intensity:
                session = None

        if session is None:
            while len(_preview_sessions) >= _MAX_PREVIEW_SESSIONS:
                oldest_key = min(
                    _preview_sessions,
                    key=lambda key: _preview_sessions[key].last_used,
                )
                _preview_sessions.pop(oldest_key, None)
            session = _PreviewSession(
                request.model,
                request.intensity,
                _load_learned_model(request.model),
            )
            _preview_sessions[session_id] = session

        session.last_used = now
        return session


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


def _render_package(request: RenderRequest, *, preview: bool) -> tuple[io.BytesIO | bytes, str]:
    duration = _PREVIEW_SECONDS if preview else request.duration
    # Allow client to request shorter preview (for quick-start first chunk)
    if preview and hasattr(request, 'preview_duration') and request.preview_duration:
        duration = min(request.preview_duration, _PREVIEW_SECONDS)
    if preview and request.session_id:
        session = _get_preview_session(request)
        chunk_index = request.chunk_index
        with session.lock:
            session.last_used = time.monotonic()
            if chunk_index is None:
                chunk_index = session.last_chunk_index + 1
            cache_key = (chunk_index, duration, request.model, request.intensity)
            if cache_key == session.cached_key and session.cached_wav is not None:
                return session.cached_wav, session.cached_filename
            if chunk_index != session.last_chunk_index + 1:
                raise HTTPException(
                    status_code=409,
                    detail="Preview chunks must be requested in sequence.",
                )

            audio, sr = _render_audio(
                request.model,
                duration,
                request.intensity,
                fast=False,
                synthesizer=session.synthesizer,
            )
            filename = _export_filename(request.model, duration, request.intensity)
            buffer = _audio_to_wav_buffer(audio, sr)
            session.last_chunk_index = chunk_index
            session.cached_key = cache_key
            session.cached_wav = buffer.getvalue()
            session.cached_filename = filename
            return session.cached_wav, filename

    # Stateless previews remain supported for older clients and API callers.
    # Playback uses the same full-resolution spectral evolution as exports.
    audio, sr = _render_audio(request.model, duration, request.intensity, fast=False)
    filename = _export_filename(request.model, duration, request.intensity)

    if not preview:
        export_path = _EXPORTS_DIR / filename
        sf.write(str(export_path), audio, sr, format="WAV", subtype=config.BIT_DEPTH)

    buffer = _audio_to_wav_buffer(audio, sr)
    return (buffer.getvalue() if preview else buffer), filename


@app.get("/")
def index():
    if not _INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Static UI not found.")
    return FileResponse(_INDEX_FILE)


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    if not _FAVICON_FILE.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(_FAVICON_FILE, media_type="image/svg+xml")


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
    wav, filename = await asyncio.to_thread(_render_package, request, preview=True)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="preview-{filename}"'},
    )


@app.post("/api/preview/close", status_code=204)
async def close_preview_session(session_id: str):
    _close_preview_session(session_id)
    return Response(status_code=204)


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
