# hyperfocus2-ocr — developer guide

## Goal

A standalone, stateless HTTP microservice that extracts the four survivor
usernames from the bottom-left HUD panel of Dead by Daylight 1280×720
stream-preview screenshots.

Target: **~0.15 s per request, 87%+ name accuracy** on CPU.

## Project structure

```
hyperfocus2-ocr/
├── app/
│   ├── __init__.py          # Package docstring
│   ├── extract.py           # Geometry constants, Detection dataclass,
│   │                          dedup, row-clustering (extract_names),
│   │                          EasyOCR wrappers, CLI, test runner
│   ├── fastocr.py           # RapidOCR-ONNX pipeline: in-process cv2,
│   │                          hybrid detect+recognise, multiprocessing,
│   │                          CLI, test runner
│   ├── engine.py            # Long-running engine: single RapidOCR instance,
│   │                          asyncio.Lock, run_in_executor
│   └── server.py            # FastAPI: POST /ocr, GET /healthz,
│                              lifespan-based engine startup, env-var config
├── testdata/                # 18 labelled screenshots (jpg + json)
├── img/                     # README illustrations (not in Docker image)
├── Dockerfile               # python:3.12-slim, uvicorn entrypoint
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Architecture

```
HTTP request (JPEG bytes)
        │
        ▼
   FastAPI (server.py)
        │
        ▼
   Engine.acquire() ── asyncio.Lock ──► run_in_executor()
        │                                    │
        │                              fastocr.process_array()
        │                              (crop → upscale → OCR → cluster)
        │                                    │
        ▼                                    ▼
   {"names": [...]} ◄──────────────────── result
```

- The engine is loaded **once** during FastAPI lifespan startup (~0.2 s).
- Access is serialised with an `asyncio.Lock` because the ONNX session is not
  thread-safe.
- The CPU-bound inference runs via `loop.run_in_executor` so the event loop
  keeps accepting connections.
- Parallelism comes from **horizontal scaling** — N containers, each with one
  engine, behind a load balancer (Traefik round-robin).

## Pipeline details

```
full 1280×720 screenshot
        │
        ▼  crop 265×420+0+300
   native panel (265×420)
        │
        ├──► enhance (contrast stretch) ──► text detection (DBNet, ONNX)
        │                                       │
        ▼                                    bounding boxes
   upscale 4× bilinear (1060×1680)
        │
        ▼  enhance (contrast stretch)
   enhanced 4× panel
        │
        ▼  crop text regions from boxes
   text crops (4×)
        │
        ▼  text recognition (CRNN, ONNX)
   (text, confidence) per box
        │
        ▼  dedup overlapping boxes
        ▼  filter noise (BOTTOM_NOISE_FRAC)
        ▼  greedy cluster into 4 rows (min_sep)
   ["Name1", "Name2", "Name3", "Name4"]
```

### Hybrid mode (default)

Detection runs on the small native panel (cheap), recognition on 4× crops
(accurate). Detection dominates cost and scales with pixel count — running it
at 1× roughly halves per-image time with no accuracy loss.

### Geometry constants (`app/extract.py`)

| Constant | Value | Meaning |
|---|---|---|
| `PANEL_X, PANEL_Y` | `0, 300` | Top-left of HUD panel in 1280×720 |
| `PANEL_W × PANEL_H` | `265 × 420` | Panel dimensions |
| `SCALE` | `4` | Upscale factor |
| `OUT_W × OUT_H` | `1060 × 1680` | 4× scaled dimensions |
| `BOTTOM_NOISE_FRAC` | `0.82` | Fraction of panel height below which detections are discarded |

If the HUD layout changes in a future DbD patch, adjust these constants.

### Clustering (`extract_names` in `app/extract.py`)

1. **Dedup** overlapping detections (keep highest-scored).
2. **Noise filter** — discard detections with `y > OUT_H × BOTTOM_NOISE_FRAC`
   (ability-bar numbers, watermark text).
3. **Anchor selection** — pick up to 4 candidates (length ≥ 3, ≥ 2 letters)
   greedily by score, enforcing minimum vertical separation `min_sep`.
4. **Row merge** — for each anchor, gather fragments within `row_half` px
   vertically, merge left-to-right.

`min_sep` and `row_half` are calibrated fractions of `OUT_H` (`/12` and `/13`
respectively) so that the clustering adapts if panel geometry changes.

## Development workflow

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No system dependencies needed — only `opencv-python`, `rapidocr-onnxruntime`,
and `onnxruntime` (ONNX models ~12 MB, no PyTorch).

### Run the server locally

```bash
OCR_LOG_LEVEL=debug uvicorn app.server:app --host 0.0.0.0 --port 8081
# or: python -m app.server
```

### Run tests

```bash
# Accuracy against testdata/ fixtures
python -m app.fastocr --test

# Single image
python -m app.fastocr --json testdata/18.jpg

# Batch
python -m app.fastocr --workers 8 --json testdata/*.jpg
```

### Add a test fixture

1. Drop a 1280×720 screenshot as `testdata/N.jpg`.
2. Create `testdata/N.json`:
   ```json
   {"survivors": ["Player1", "Player2", "Player3", "Player4"]}
   ```
3. Run `python -m app.fastocr --test` and verify the new image is picked up.

### Smoke test

```bash
curl -s http://localhost:8081/healthz
curl -s --data-binary @testdata/18.jpg \
     -H 'Content-Type: image/jpeg' \
     http://localhost:8081/ocr | jq .
```

### Build & run Docker

```bash
docker build -t hyperfocus2-ocr .
docker run -p 8081:8081 -e OCR_THREADS=1 hyperfocus2-ocr
```

The Docker image excludes `testdata/`, `img/`, `*.md`, and `.git/`
(`.dockerignore`).

## Accuracy

Current baseline: **87%** (63/72 names across 18 fixtures, RapidOCR hybrid
mode, 265×420 px crop). Evaluation uses edit distance with tolerance
`max(2, len(name)//5)`.

Known failure modes:
- **Undetected names** — the DBNet detector occasionally misses names on
  very faint or heavily obscured panels (~2-3 per 72).
- **Garbled characters** — CRNN hallucinates on borderline pixels, producing
  1-3 char errors (~4-5 per 72). Most common with Cyrillic glyphs.
- **Watermark interference** — the "CREATOR DEAD BY DAYLIGHT" text near the
  panel bottom can leak through the noise filter on some screenshots.

## Notes

- ONNX inference on this workload is **memory-bandwidth bound** — 8 processes
  on 8 cores only give ~2× throughput over 1 process. For higher throughput,
  use a GPU with CUDA Execution Provider.
- `--threads` / `OCR_THREADS` controls ONNX intra-op threads. Setting it > 1
  in a single-process server does not help (the session is serialised by the
  lock). It's only useful in the multiprocessing CLI path (`--workers N`).
- The angle classifier is disabled — HUD name rows are always horizontal,
  skipping the extra model pass saves ~30 ms per crop.

## Related repos

| Repo | Role |
|---|---|
| [`rofleksey/hyperfocus2`](https://github.com/rofleksey/hyperfocus2) | Main app: Twitch poller + web service + Postgres. Calls this OCR service. |
| [`rofleksey/yandex-sites-config`](https://github.com/rofleksey/yandex-sites-config) | Production Docker Compose deployment (Traefik + hyperfocus2 + this OCR service). |
| `ocr_test` (archived) | Original R&D scratchpad — replaced by this repo. |
