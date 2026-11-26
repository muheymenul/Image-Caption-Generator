"""
app/api.py
FastAPI backend for the Image Caption Generator.

Run from the project root:
    uvicorn app.api:app --host 127.0.0.1 --port 8000
Web UI:      http://127.0.0.1:8000/
Swagger UI:  http://127.0.0.1:8000/docs
"""
import io
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

# Make "src" importable no matter where uvicorn is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import CaptionGenerator  # noqa: E402

# ---------------- Settings (override with environment variables) ----------------
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(PROJECT_ROOT, "saved_models"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "MPO"}   # MPO = some phone-camera JPEGs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("caption_api")

_state = {}
_inference_lock = threading.Lock()


# ---------------- Response schemas ----------------
class CaptionCandidate(BaseModel):
    caption: str = Field(..., examples=["a black dog is running through the water"])
    score: float = Field(..., description="Length-normalized log-probability (higher is better)")


class CaptionResponse(BaseModel):
    caption: str
    score: float
    alternatives: List[CaptionCandidate]
    beam_width: int
    image_size: List[int] = Field(..., description="Original [width, height]")
    inference_time_ms: float
    filename: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    trained_on: Optional[str] = None
    vocab_size: Optional[int] = None
    max_caption_len: Optional[int] = None


# ---------------- Startup / shutdown ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading caption model from %s ...", MODEL_DIR)
    t0 = time.perf_counter()
    generator = CaptionGenerator(MODEL_DIR, load_cnn=True)
    # Warm-up: trace tf.functions once so the first real request is fast
    dummy = np.zeros((generator.image_size, generator.image_size, 3), dtype=np.uint8)
    generator.caption(dummy, beam_width=1)
    _state["generator"] = generator
    logger.info("Model ready in %.1fs (trained on: %s)", time.perf_counter() - t0, generator.trained_on)
    logger.info("Open the web UI at http://127.0.0.1:8000/")
    yield
    _state.clear()
    logger.info("Model unloaded.")


app = FastAPI(
    title="Image Caption Generator API",
    version="1.1.0",
    description="EfficientNetB3 encoder + Transformer decoder with Beam Search. "
                "Upload an image to receive an English caption. Web UI is served at `/`.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------- Helpers ----------------
def decode_upload(data: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decode bytes -> RGB uint8 array, fixing phone EXIF rotation."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the file as an image.")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported format '{fmt}'. Allowed: JPEG, PNG, WebP, BMP, GIF.")
    return np.asarray(img), img.size


def get_generator() -> CaptionGenerator:
    generator = _state.get("generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return generator


# ---------------- Routes ----------------
@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    generator = _state.get("generator")
    if generator is None:
        return HealthResponse(status="loading", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        trained_on=generator.trained_on,
        vocab_size=len(generator.vocab),
        max_caption_len=generator.vocab.max_len,
    )


@app.post("/caption", response_model=CaptionResponse, tags=["Captioning"],
          summary="Generate a caption for an uploaded image")
async def caption_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, BMP, GIF)"),
    beam_width: int = Query(3, ge=1, le=10, description="1 = greedy decoding"),
    num_alternatives: int = Query(3, ge=1, le=10, description="How many ranked captions to return"),
):
    generator = get_generator()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB:.0f} MB.")

    image, (width, height) = decode_upload(data)

    def _run():
        with _inference_lock:
            return generator.caption(image, beam_width=beam_width)

    t0 = time.perf_counter()
    try:
        results = await run_in_threadpool(_run)
    except Exception:
        logger.exception("Inference failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Inference failed. Check the server logs.")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    best = results[0]
    logger.info("%s -> '%s' (%.0f ms)", file.filename, best["caption"], elapsed_ms)
    return CaptionResponse(
        caption=best["caption"],
        score=best["score"],
        alternatives=[CaptionCandidate(**r) for r in results[:num_alternatives]],
        beam_width=beam_width,
        image_size=[width, height],
        inference_time_ms=round(elapsed_ms, 1),
        filename=file.filename,
    )
