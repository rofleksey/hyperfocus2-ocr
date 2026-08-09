"""Single preloaded OCR engine for one service instance.

One RapidOCR engine is built at startup and reused for every request. Inference
is serialized with a lock because an ONNX ``InferenceSession`` is not safe for
concurrent use from multiple threads; horizontal throughput comes from running
multiple instances of this service behind a load balancer (e.g. Traefik), not
from concurrency inside a single process.

CPU-bound inference is dispatched to a thread via ``run_in_executor`` so the
asyncio event loop stays free to accept new connections while the (locked)
inference runs.
"""
from __future__ import annotations

import asyncio
import logging
import os

import cv2
import numpy as np

from .fastocr import build_engine, process_array

log = logging.getLogger("ocr.engine")


class Engine:
    """One preloaded RapidOCR engine guarded by a lock."""

    def __init__(self, threads: int, hybrid: bool):
        self.threads = max(1, threads)
        self.hybrid = hybrid
        self._engine = None
        self._lock = asyncio.Lock()

    def start(self) -> None:
        """Load the engine synchronously at startup (~0.2 s ONNX model load)."""
        log.info("loading ocr engine (threads=%d hybrid=%s)", self.threads, self.hybrid)
        self._engine = build_engine(threads=self.threads)
        log.info("ocr engine ready (pid=%s)", os.getpid())

    async def infer(self, jpeg: bytes) -> list[str]:
        """Decode raw JPEG bytes and run OCR. Serialized on a single engine."""
        if self._engine is None:
            raise RuntimeError("ocr engine not loaded")
        async with self._lock:
            return await asyncio.get_running_loop().run_in_executor(
                None, self._infer_sync, jpeg)

    def _infer_sync(self, jpeg: bytes) -> list[str]:
        im = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return []
        return process_array(self._engine, im, hybrid=self.hybrid)

    def shutdown(self) -> None:
        self._engine = None
        log.info("ocr engine released")
