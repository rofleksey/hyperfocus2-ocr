"""hyperfocus2-ocr: a long-running microservice that preloads a single
RapidOCR / PaddleOCR-ONNX engine and extracts Dead by Daylight survivor
nicknames from preview screenshots, one image per request.

The OCR pipeline itself lives in :mod:`app.fastocr` (moved verbatim from the
monolith) and the shared geometry / post-processing helpers in
:mod:`app.extract`. :mod:`app.engine` wraps a single in-process engine (guarded
by a lock) so the ONNX model is loaded exactly once at startup, and
:mod:`app.server` exposes it over HTTP via FastAPI. Horizontal throughput is
achieved by running multiple instances behind a load balancer (e.g. Traefik).
"""
