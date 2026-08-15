#!/usr/bin/env python3
"""Fast DbD survivor-name extraction (RapidOCR / PaddleOCR-ONNX).

~110x higher throughput and ~11pp more accurate than the original EasyOCR
pipeline (8.4 s/image -> ~0.08 s/image effective; 77% -> 90% name accuracy):

  * in-process cv2 crop + 4x bilinear upscale + contrast stretch (no ffmpeg /
    ImageMagick subprocesses — those alone cost ~0.5 s/image and are hostile to
    parallelism).
  * RapidOCR runs PaddleOCR's PP-OCRv3 detection (DBNet) + recognition (CRNN)
    via ONNX Runtime. On CPU this is far lighter than EasyOCR's CRAFT detector
    and recognises the small white-on-dark name glyphs more accurately. ONNX
    models also load in ~0.2 s (vs ~4 s for torch+EasyOCR).
  * Hybrid mode detects text boxes on the native panel and recognises on 4x
    upscaled crops. Only the box regions are upscaled to 4x (the whole-panel
    upscale cost ~100 ms at 1080p for pixels that are never recognised), and
    noise boxes below the HUD rows are filtered before recognition.
    2500 screenshots process in ~7 min on an 8-core CPU via the multiprocessing
    batch runner (~2x over single-process; the workload is memory-bandwidth
    bound).
  * The existing dedup / row-clustering post-processing is reused unchanged.

Usage:
    python -m app.fastocr testdata/1.jpg               # names, one per line
    python -m app.fastocr --json testdata/*.jpg        # {"image":..,"names":[..]}
    python -m app.fastocr --workers 8 --json *.jpg     # parallel (2500 in ~7 min)
    python -m app.fastocr --test                       # accuracy vs testdata/*.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

try:  # when imported as part of the `app` package (server / `python -m app.fastocr`)
    from .extract import (
        Detection, Geometry, extract_names, load_fixtures, check_placeholders,
        BOTTOM_NOISE_FRAC, SCALE,
    )
except ImportError:  # when run as a standalone script
    from extract import (
        Detection, Geometry, extract_names, load_fixtures, check_placeholders,
        BOTTOM_NOISE_FRAC, SCALE,
    )

# NOTE: ONNX thread count is set per-worker in _init_worker(), not globally, so
# the single-process path keeps using all cores (which is ~2.5x faster here).

_ORT_PATCHED = False


def _patch_ort_threads(n: int):
    """Force intra/inter op threads on every ONNX session RapidOCR creates.
    onnxruntime ignores OMP_NUM_THREADS for its own pool, so we must set it on
    SessionOptions. Idempotent."""
    global _ORT_PATCHED
    if _ORT_PATCHED:
        return
    import onnxruntime as ort
    import rapidocr_onnxruntime as R

    def _init(self, config):
        so = ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        from onnxruntime import InferenceSession
        eps = [('CPUExecutionProvider',
                {'arena_extend_strategy': 'kSameAsRequested'})]
        self.session = InferenceSession(config['model_path'],
                                        sess_options=so, providers=eps)

    R.utils.OrtInferSession.__init__ = _init
    _ORT_PATCHED = True


def build_engine(threads: int | None = None, use_angle_cls: bool = False):
    if threads is not None:
        _patch_ort_threads(threads)
    from rapidocr_onnxruntime import RapidOCR
    # Angle classification is pointless here: the HUD name rows are always
    # horizontal, so skip the extra orientation model pass per text crop.
    # Detection keeps the default min/736 internal input size — shrinking it
    # costs accuracy (measured -2..-3pp at 1080p).
    return RapidOCR(use_angle_cls=use_angle_cls)


def enhance(bgr: np.ndarray) -> np.ndarray:
    """Contrast stretch emulating `magick -level 0,80%,1.15 -contrast`."""
    lo, hi = 0.0, 255.0 * 0.8
    norm = np.clip((bgr.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255)
    return (np.power(norm / 255.0, 1.0 / 1.15) * 255.0).astype(np.uint8)


def preprocess_array(im: np.ndarray) -> tuple[np.ndarray, Geometry]:
    """Crop the HUD panel, 4x upscale and contrast-stretch a decoded image.

    The panel geometry adapts to the image resolution (720p baseline scaled by
    height/720), so 1280x720 and 1920x1080 previews both work.
    """
    geo = Geometry.for_size(im.shape[0], im.shape[1])
    crop = im[geo.y:geo.y + geo.h, geo.x:geo.x + geo.w]
    up = cv2.resize(crop, (geo.out_w, geo.out_h), interpolation=cv2.INTER_LINEAR)
    return enhance(up), geo


def preprocess(img_path: str) -> tuple[np.ndarray, Geometry]:
    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(img_path)
    return preprocess_array(im)


def process_array(engine, im: np.ndarray, hybrid: bool = False) -> list[str]:
    """Run the OCR pipeline on a decoded BGR image (no disk I/O)."""
    if hybrid:
        return _process_hybrid_array(engine, im)
    rec, geo = preprocess_array(im)
    result, _elapse = engine(rec)
    dets: list[Detection] = []
    if result:
        for box, text, score in result:
            ys = [p[1] for p in box]; xs = [p[0] for p in box]
            y = int(min(ys)); x = int(min(xs))
            if y > geo.out_h * BOTTOM_NOISE_FRAC:
                continue
            dets.append(Detection(
                y=y, x=x,
                w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys)),
                conf=float(score), text=text.strip(),
            ))
    return extract_names(dets, geo.out_h)


def process(engine, img_path: str, hybrid: bool = False) -> list[str]:
    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(img_path)
    return process_array(engine, im, hybrid=hybrid)


def _rotate_crop(img: np.ndarray, points) -> np.ndarray:
    """Perspective-crop a box from ``img`` (same math as RapidOCR's
    ``get_rotate_crop_image``: INTER_CUBIC warp, BORDER_REPLICATE, rotate tall
    crops by 90 deg)."""
    w = int(max(np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(img, M, (w, h),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


def _box_crop_4x(crop: np.ndarray, box) -> np.ndarray:
    """Build the 4x enhanced recognition crop for one native-panel box.

    Only the box region is upscaled — upscaling the whole panel costs ~100 ms
    at 1080p for pixels that are never recognised. The result is the same crop
    the old whole-panel path produced: the region content is the exact 4x
    bilinear upscale of the panel crop (integer scale), enhanced with the same
    elementwise stretch, then warped at 4x resolution (max |diff| vs the old
    path measured <= 1 gray level)."""
    x0 = max(int(np.floor(min(p[0] for p in box))) - 1, 0)
    y0 = max(int(np.floor(min(p[1] for p in box))) - 1, 0)
    x1 = min(int(np.ceil(max(p[0] for p in box))) + 1, crop.shape[1])
    y1 = min(int(np.ceil(max(p[1] for p in box))) + 1, crop.shape[0])
    reg = crop[y0:y1, x0:x1]
    up = cv2.resize(reg, ((x1 - x0) * SCALE, (y1 - y0) * SCALE),
                    interpolation=cv2.INTER_LINEAR)
    shifted = (box - np.array([x0, y0], dtype=np.float32)) * SCALE
    return _rotate_crop(enhance(up), shifted)


def _process_hybrid_array(engine, im: np.ndarray) -> list[str]:
    """Detect text boxes on the small native panel (cheap), then recognise on
    4x enhanced crops (accurate). Boxes below BOTTOM_NOISE_FRAC are filtered
    before recognition so the CRNN never sees ability-bar numbers or
    watermark text."""
    geo = Geometry.for_size(im.shape[0], im.shape[1])
    crop = im[geo.y:geo.y + geo.h, geo.x:geo.x + geo.w]
    det_img = enhance(crop)

    boxes, _ = engine.text_detector(det_img)
    if boxes is None or len(boxes) < 1:
        return []
    boxes = engine.sorted_boxes(boxes)

    max_y = geo.out_h * BOTTOM_NOISE_FRAC
    kept4: list = []
    crops: list = []
    for b in boxes:
        b4 = b * SCALE
        if min(p[1] for p in b4) > max_y:
            continue
        kept4.append(b4)
        crops.append(_box_crop_4x(crop, b))
    if not crops:
        return []
    rec_res, _ = engine.text_recognizer(crops)

    dets: list[Detection] = []
    for b, (text, score) in zip(kept4, rec_res):
        ys = [p[1] for p in b]; xs = [p[0] for p in b]
        y = int(min(ys)); x = int(min(xs))
        if y > geo.out_h * BOTTOM_NOISE_FRAC:
            continue
        dets.append(Detection(
            y=y, x=x, w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys)),
            conf=float(score), text=text.strip(),
        ))
    return extract_names(dets, geo.out_h)


# --- multiprocessing --------------------------------------------------------
_WORKER = {}


def _init_worker(threads: int, hybrid: bool = False):
    # Cap ONNX threads inside each worker to avoid oversubscription when many
    # processes run concurrently. OMP_NUM_THREADS alone is insufficient because
    # onnxruntime runs its own pool; we also patch the SessionOptions.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    _patch_ort_threads(threads)
    _WORKER["engine"] = build_engine()
    _WORKER["hybrid"] = hybrid


def _work(img_path: str) -> tuple[str, list[str]]:
    return img_path, process(_WORKER["engine"], img_path, hybrid=_WORKER.get("hybrid", False))


def process_many(img_paths: list[str], workers: int, threads: int = 1,
                 hybrid: bool = False) -> list[tuple[str, list[str]]]:
    if workers <= 1:
        engine = build_engine()
        return [(p, process(engine, p, hybrid=hybrid)) for p in img_paths]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(threads, hybrid)) as pool:
        return pool.map(_work, img_paths)


# --- accuracy ---------------------------------------------------------------
def _edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _best_match(expected: str, candidates: list[str]) -> int:
    exp = expected.lower().replace(" ", "").rstrip(".")
    tol = max(2, len(exp) // 5)
    for c in candidates:
        if _edit_distance(exp, c.lower().replace(" ", "")) <= tol:
            return 1
    return 0


def run_test(workers: int, hybrid: bool = True) -> int:
    data = Path(__file__).parent.parent / "testdata"
    pairs = load_fixtures(data)
    if not pairs:
        print("no fixtures found under testdata/")
        return 0
    check_placeholders(pairs)
    results = process_many([str(p) for _, p, _ in pairs], workers, hybrid=hybrid)
    rmap = {p: n for p, n in results}
    totals: dict[str, list[int]] = {}
    for label, img, jf in pairs:
        expected = json.loads(jf.read_text())["survivors"]
        got = rmap[str(img)]
        totals.setdefault(label, [0, 0])
        for exp in expected:
            totals[label][0] += 1
            totals[label][1] += _best_match(exp, got)
        print(f"### [{label}] {img.name}")
        print(f"    got={got}")
    for label, (t, m) in totals.items():
        pct = 100 * m // t if t else 0
        print(f"=== {label}: {m}/{t} matched ({pct}%) ===")
    grand = [sum(c[i] for c in totals.values()) for i in range(2)]
    pct = 100 * grand[1] // grand[0] if grand[0] else 0
    print(f"=== total: {grand[1]}/{grand[0]} matched ({pct}%) ===")
    return pct


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1,
                    help="ONNX threads per worker (multiprocessing only)")
    ap.add_argument("--full", action="store_true",
                    help="detect AND recognise at 4x (more conservative). Both "
                         "modes score ~90%% on the test set; default hybrid is "
                         "faster and meets the batch-runner target.")
    args = ap.parse_args()

    hybrid = not args.full

    if args.test:
        run_test(args.workers, hybrid=hybrid)
        return
    if not args.images:
        ap.error("provide image paths or use --test")

    results = process_many(args.images, args.workers, args.threads, hybrid=hybrid)
    for img, names in results:
        if args.json:
            print(json.dumps({"image": img, "names": names}))
        else:
            for n in names:
                print(n)


if __name__ == "__main__":
    main()
