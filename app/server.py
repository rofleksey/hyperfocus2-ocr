"""FastAPI application exposing a single preloaded OCR engine over HTTP.

The engine is built once during the lifespan startup (so the ONNX model loads a
single time, ~0.2 s) and reused for every request. The service is then ready to
extract survivor nicknames from any number of incoming images without ever
reloading the model. Scale horizontally by running multiple instances of this
container behind a load balancer (e.g. Traefik).

Endpoints:
    POST /ocr       raw JPEG body  ->  {"names": ["...", ...]}
    GET  /healthz   ->  {"status": "ok", "hybrid": true}

Configuration is via environment variables (12-factor, container-friendly):

    OCR_HOST        listen host        (default 0.0.0.0)
    OCR_PORT        listen port        (default 8081)
    OCR_THREADS     ONNX threads/engine(default 1)
    OCR_HYBRID      hybrid detect mode (default true)
    OCR_MAX_BYTES   request body cap   (default 8 MiB)
    OCR_LOG_LEVEL   log level          (default info)

Run with either:
    uvicorn app.server:app --host 0.0.0.0 --port 8081
    python -m app.server
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .engine import Engine

log = logging.getLogger("ocr.server")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


# --- configuration ---------------------------------------------------------
HOST = _env("OCR_HOST", "0.0.0.0")
PORT = _env_int("OCR_PORT", 8081)
THREADS = _env_int("OCR_THREADS", 1)
HYBRID = _env_bool("OCR_HYBRID", True)
MAX_BYTES = _env_int("OCR_MAX_BYTES", 8 * 1024 * 1024)
LOG_LEVEL = _env("OCR_LOG_LEVEL", "info")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

# A single preloaded engine, created on startup and destroyed on shutdown.
engine = Engine(threads=THREADS, hybrid=HYBRID)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Preload the engine once, serve forever, then release it."""
    log.info("ocr service starting (host=%s port=%d threads=%d hybrid=%s)",
             HOST, PORT, THREADS, HYBRID)
    engine.start()  # loads the ONNX model eagerly before serving traffic
    log.info("ocr service ready")
    try:
        yield
    finally:
        engine.shutdown()
        log.info("ocr service stopped")


app = FastAPI(title="hyperfocus2-ocr", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "hybrid": HYBRID}


@app.post("/ocr")
async def ocr(request: Request) -> JSONResponse:
    """Extract survivor nicknames from a single JPEG image sent as the body."""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_BYTES:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    f"image exceeds {MAX_BYTES} bytes")
        except ValueError:
            pass

    body = await request.body()
    if not body:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty body")
    if len(body) > MAX_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"image exceeds {MAX_BYTES} bytes")
    # Reject obvious non-JPEG payloads early (preview captures are always JPEG).
    if body[:2] != b"\xff\xd8":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            "expected a JPEG image body")

    try:
        names = await engine.infer(body)
    except Exception as exc:  # noqa: BLE001 - surface any inference failure as 5xx
        log.exception("ocr inference failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"ocr inference failed: {exc}") from exc

    return JSONResponse({"names": names})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host=HOST, port=PORT, log_level=LOG_LEVEL)
