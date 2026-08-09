# hyperfocus2-ocr

A standalone microservice that extracts **Dead by Daylight survivor nicknames**
from 1280×720 stream-preview screenshots. It preloads a single
RapidOCR / PaddleOCR-ONNX engine **once** at startup and then answers one
image per request over HTTP, so the (relatively expensive) ONNX model load
never happens per request.

This is the OCR half split out of the
[`hyperfocus2`](https://github.com/rofleksey/hyperfocus2) monolith. The OCR
pipeline itself (`app/extract.py`, `app/fastocr.py`) was moved here verbatim
from that project; `app/engine.py` + `app/server.py` wrap it in a long-running
FastAPI service.

## How it works

- Each instance loads **one** `RapidOCR` engine during FastAPI lifespan startup
  (~0.2 s) and keeps it resident for the life of the process.
- `POST /ocr` receives a raw JPEG body, decodes it in-memory with OpenCV, and
  runs inference on the preloaded engine. Access is serialized with a lock
  because an ONNX session is not safe for concurrent use from multiple threads;
  the CPU-bound work runs in a thread via `loop.run_in_executor` so the event
  loop keeps accepting connections.
- **Scale horizontally** by running N instances of this container behind a load
  balancer (e.g. Traefik round-robin). Parallelism comes from many containers,
  not from many engines inside one process — keeping each instance small,
  stateless and cheap to size.

## API

### `POST /ocr`
**Body:** the raw bytes of a JPEG preview (`Content-Type: image/jpeg`).
**Response:** `200`
```json
{ "names": ["SurvivorOne", "PlayerTTV", "..."] }
```
Errors: `400` empty body, `415` not a JPEG, `413` too large, `502` inference
failure.

### `GET /healthz`
```json
{ "status": "ok", "hybrid": true }
```

## Configuration (environment variables)

| Variable       | Default   | Meaning                                            |
|----------------|-----------|----------------------------------------------------|
| `OCR_HOST`     | `0.0.0.0` | listen host                                        |
| `OCR_PORT`     | `8081`    | listen port                                        |
| `OCR_THREADS`  | `1`       | ONNX intra-op threads for the engine               |
| `OCR_HYBRID`   | `true`    | hybrid detect-on-native / recognise-on-4x mode     |
| `OCR_MAX_BYTES`| `8388608` | request body cap (bytes)                           |
| `OCR_LOG_LEVEL`| `info`    | `debug` / `info` / `warning` / `error`             |

Tune throughput by running more replicas (one engine ≈ one CPU's worth of
inference). The calling service
([hyperfocus2](https://github.com/rofleksey/hyperfocus2)) controls how many
in-flight requests it makes via its own `ocr.workers` config.

## Running

### Local (venv)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
OCR_THREADS=1 uvicorn app.server:app --host 0.0.0.0 --port 8081
# or:  python -m app.server
```

### Docker (single instance)
```bash
docker build -t hyperfocus2-ocr .
docker run -p 8081:8081 hyperfocus2-ocr
```

### Docker + Traefik (N replicas, round-robin)
Run N replicas behind Traefik, e.g. via a Docker Compose `scale` or a Swarm/K8s
deployment, and point `hyperfocus2`'s `ocr.api_url` at the Traefik frontend:

```yaml
# docker-compose.yml (sketch)
services:
  ocr:
    build: .
    deploy:
      replicas: 4
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ocr.rule=PathPrefix(`/ocr`)"
      - "traefik.http.services.ocr.loadBalancer.server.port=8081"
  traefik:
    image: traefik:v3
    command: --providers.docker --providers.docker.exposedbydefault=false
    ports: ["8081:8081"]
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]
```
Then set `ocr.api_url: "http://traefik:8081"` in `hyperfocus2`'s config.

### Smoke test
```bash
curl -s http://localhost:8081/healthz
curl -s --data-binary @preview.jpg -H 'Content-Type: image/jpeg' http://localhost:8081/ocr
```

## Accuracy harness (local dev)

The original CLI was preserved for measuring accuracy against labelled fixtures.
Place `1.jpg` / `1.json` (with `{"survivors": [...]}`) pairs under `app/data/`
and run:

```bash
python -m app.fastocr --test            # accuracy report
python -m app.fastocr --json data/*.jpg # batch CLI (one-off, unrelated to the server)
```
