# hyperfocus2-ocr — developer guide

## Goal

A standalone, stateless HTTP microservice that extracts the four survivor
usernames from the bottom-left HUD panel of Dead by Daylight stream-preview
screenshots. 1080p (1920×1080) previews are the primary input; 720p
(1280×720) is fully supported as a fallback.

Target: **~0.15 s per request, 90%+ name accuracy** on CPU.

## Project structure

```
hyperfocus2-ocr/
├── app/
│   ├── __init__.py          # Package docstring
│   ├── extract.py           # Geometry (resolution-adaptive), Detection
│   │                          dataclass, dedup, row-clustering
│   │                          (extract_names), EasyOCR wrappers, CLI,
│   │                          test runner, fixture loading
│   ├── fastocr.py           # RapidOCR-ONNX pipeline: in-process cv2,
│   │                          hybrid detect+recognise, multiprocessing,
│   │                          CLI, test runner
│   ├── engine.py            # Long-running engine: single RapidOCR instance,
│   │                          asyncio.Lock, run_in_executor
│   └── server.py            # FastAPI: POST /ocr, GET /healthz,
│                              lifespan-based engine startup, env-var config
├── testdata/
│   ├── 720p/                # 18 labelled 720p screenshots (jpg + json)
│   └── 1080p/               # 18 labelled 1080p screenshots (jpg + json)
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
full screenshot (1920x1080 or 1280x720)
        │
        ▼  crop HUD panel (resolution-adaptive, see Geometry)
   native panel
        │
        ├──► enhance (contrast stretch) ──► text detection (DBNet, ONNX)
        │                                       │
        ▼                                    bounding boxes
   upscale 4× bilinear
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

### Geometry (`app/extract.py`)

The HUD is laid out proportionally to the stream resolution, so panel
geometry is derived from the measured 1280×720 baseline scaled by
`height/720` (`Geometry.for_size`). Twitch previews arrive as 1920×1080
(factor 1.5) or 1280×720 (factor 1.0); both — and any other size — run
through the same pipeline.

| Constant | Value | Meaning |
|---|---|---|
| `PANEL_X, PANEL_Y` | `0, 300` | Top-left of HUD panel in 1280×720 |
| `PANEL_W × PANEL_H` | `265 × 420` | Panel dimensions (720p baseline) |
| `PANEL_TOP_MARGIN` | `28` | Extra headroom above the panel top (720p units). The panel top varies between streamers (HUD scale setting), the panel always reaches the bottom frame edge. |
| `SCALE` | `4` | Upscale factor |
| `BOTTOM_NOISE_FRAC` | `0.78` | Fraction of crop height below which detections are discarded |

Effective crops: 1280×720 → `265×448+0+272`; 1920×1080 → `398×672+0+408`.
If the HUD layout changes in a future DbD patch, adjust these constants.

### Clustering (`extract_names` in `app/extract.py`)

1. **Dedup** overlapping detections (keep highest-scored).
2. **Noise filter** — discard detections with `y > out_h × BOTTOM_NOISE_FRAC`
   (ability-bar numbers, watermark text).
3. **Anchor selection** — pick up to 4 candidates greedily by score,
   enforcing minimum vertical separation `min_sep`. A candidate must look
   like a name: ≥3 chars with ≥1 letter, OR any length with conf ≥ 0.45
   (recovers very short names like "m").
4. **Row merge** — for each anchor, gather fragments within `row_half` px
   vertically, merge left-to-right.

`min_sep` and `row_half` are calibrated fractions of `out_h` (`/14` each) so
that the clustering adapts to panel geometry and to compact layouts (e.g.
character-select screens with ~200 px rows at 4×).

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

1. Drop a screenshot as `testdata/<res>/N.jpg` (`res` = `720p` or `1080p`).
2. Create `testdata/<res>/N.json`:
   ```json
   {"survivors": ["Player1", "Player2", "Player3", "Player4"]}
   ```
3. Run `python -m app.fastocr --test` and verify the new image is picked up.

The test runner refuses to run if any `XXXX` placeholder remains in a JSON.

### Smoke test

```bash
curl -s http://localhost:8081/healthz
curl -s --data-binary @testdata/1080p/1.jpg \
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

Current baseline: **1080p 93%** (67/72) and **720p 87%** (63/72), RapidOCR
hybrid mode, resolution-adaptive crop. Evaluation uses edit distance with
tolerance `max(2, len(name)//5)`.

Known failure modes:
- **Faint panels** — names over bright game areas lose contrast and read as
  fragments or garbage (~1-2 per 72).
- **Garbled characters** — CRNN hallucinates on borderline pixels, producing
  1-3 char errors (~2-3 per 72). Most common with Cyrillic glyphs, which
  degrade into visually similar Latin text (e.g. "Штора" → "TUropa").
- **Overlay interference** — streamer overlays or hooked-survivor nameplates
  near the top of the panel can displace or merge with a name row.

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
