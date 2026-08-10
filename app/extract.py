#!/usr/bin/env python3
"""Extract survivor usernames from Dead by Daylight (1280x720) screenshots.

Pipeline (quality first, then speed):
  1. ffmpeg crops the bottom-left HUD panel and upscales it 4x (Lanczos). The
     ~14px name glyphs become ~56px, which EasyOCR recognises reliably.
  2. ImageMagick (magick) stretches contrast so faint white-on-dark names pop.
     The HUD panel is a translucent dark overlay; behind bright game areas the
     text loses contrast.
  3. EasyOCR (English + Russian) detects and recognises text. It handles Latin +
     Cyrillic mixed usernames and uneven backgrounds far better than tesseract
     for this kind of small scene text. By default two passes (plain + enhanced)
     are merged so each name row keeps its best read.
  4. Detections are deduplicated, clustered by vertical position into the four
     survivor rows, and noise (ability-bar numbers, avatar fragments, UI
     overlays) is filtered out.

Setup (one time):
    python3 -m venv .venv && source .venv/bin/activate
    pip install easyocr          # pulls in torch, ~800 MB

Usage:
    python extract.py data/1.jpg              # one image -> names on stdout
    python extract.py --json data/*.jpg       # JSON per image
    python extract.py --fast data/*.jpg       # single pass (~2x faster)
    python extract.py --test                  # accuracy report vs data/*.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# --- panel geometry (measured on 1280x720 screenshots) -----------------------
PANEL_X, PANEL_Y = 0, 300
PANEL_W, PANEL_H = 265, 420
SCALE = 4
# Scaled dimensions.
OUT_W = PANEL_W * SCALE
OUT_H = PANEL_H * SCALE
# Ability-bar numbers / icons sit at the very bottom of the crop; anything below
# this y-fraction is discarded as noise.
BOTTOM_NOISE_FRAC = 0.82


@dataclass
class Detection:
    y: int
    x: int
    w: int
    h: int
    conf: float
    text: str


def ffmpeg_crop_scale(src: str, dst: str) -> None:
    """Crop the HUD panel and upscale it with ffmpeg."""
    vf = (
        f"crop={PANEL_W}:{PANEL_H}:{PANEL_X}:{PANEL_Y},"
        f"scale={OUT_W}:{OUT_H}:flags=lanczos"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vf", vf, "-frames:v", "1", dst,
         "-loglevel", "error"],
        check=True, capture_output=True,
    )


def magick_enhance(src: str, dst: str) -> None:
    """Boost white-text contrast with ImageMagick.

    The HUD panel is a translucent dark overlay; behind bright game areas the
    white name text loses contrast. Stretching the histogram and adding contrast
    makes faint names pop without the artefacts that hard binarisation
    (grayscale + threshold, which helped tesseract) introduces for EasyOCR —
    EasyOCR relies on the colour channels, so we keep them.
    """
    subprocess.run(
        ["magick", src,
         "-level", "0,80%,1.15",
         "-contrast",
         dst],
        check=True, capture_output=True,
    )


def ocr_image(reader, path: str) -> list[Detection]:
    """Run EasyOCR on a preprocessed image."""
    raw = reader.readtext(
        path,
        detail=1,
        paragraph=False,
        text_threshold=0.3,
        low_text=0.3,
        link_threshold=0.3,
        add_margin=0.05,
        slope_ths=0.5,
        width_ths=0.5,
    )
    dets = []
    for bbox, text, conf in raw:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        dets.append(Detection(
            y=int(min(ys)), x=int(min(xs)),
            w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys)),
            conf=float(conf), text=text.strip(),
        ))
    return dets


def _has_letters(text: str) -> bool:
    """True if the token has >=2 letters (Latin or Cyrillic) — filters pure
    noise like '5', '???', '|'."""
    n = sum(1 for c in text if c.isalpha())
    return n >= 2


def _score(d: Detection) -> float:
    """Confidence weighted by text length; penalise non-name-like tokens."""
    s = d.conf * math.sqrt(max(len(d.text), 1))
    if not _has_letters(d.text):
        s *= 0.2
    return s


def _overlap(a: Detection, b: Detection) -> bool:
    """True if two detections occupy roughly the same spot (same row + column)."""
    y_close = abs(a.y - b.y) <= max(a.h, b.h) * 0.6
    x_close = abs((a.x + a.w / 2) - (b.x + b.w / 2)) <= max(a.w, b.w) * 0.7
    return y_close and x_close


def _dedup(dets: list[Detection]) -> list[Detection]:
    """Remove near-duplicate detections (from multi-pass merging), keeping the
    highest-scored copy of each."""
    ordered = sorted(dets, key=_score, reverse=True)
    kept: list[Detection] = []
    for d in ordered:
        if any(_overlap(d, k) for k in kept):
            continue
        kept.append(d)
    return kept


def extract_names(dets: list[Detection]) -> list[str]:
    """Pick the four survivor rows and reconstruct each name.

    Strategy: deduplicate overlapping detections (from multi-pass merging), then
    greedily take the highest-scored detections while enforcing a minimum
    vertical separation (so the four rows are distinct), then for each chosen row
    merge any neighbouring fragments that belong to the same name.
    """
    dets = _dedup(dets)
    max_y = int(OUT_H * BOTTOM_NOISE_FRAC)
    # Anchors must look like real names: >=3 chars and contain letters. Short
    # fragments (single Cyrillic chars, numbers) are only merged into rows, never
    # chosen as row representatives.
    cands = [d for d in dets if d.y < max_y and len(d.text) >= 1]
    anchors_pool = [d for d in cands if len(d.text) >= 3 and _has_letters(d.text)]

    # Row spacing is roughly constant in absolute pixels (~160-180px at 4x).
    # Use a flat fraction of OUT_H that stays under the minimum observed gap.
    min_sep = OUT_H / 12  # ~100-140px

    # Greedily pick anchor detections: highest score first, skipping any that are
    # too close vertically to an already-chosen anchor.
    ranked = sorted(anchors_pool, key=_score, reverse=True)
    anchors: list[Detection] = []
    for d in ranked:
        if all(abs(d.y - a.y) >= min_sep for a in anchors):
            anchors.append(d)
        if len(anchors) >= 4:
            break
    if not anchors:
        return []

    # For each anchor row, gather fragments within half the row height and merge
    # them left-to-right. Keep name-like fragments, drop obvious noise.
    row_half = OUT_H / 13
    anchors.sort(key=lambda d: d.y)
    names = []
    for anchor in anchors:
        row = [d for d in cands if abs(d.y - anchor.y) <= row_half]
        named = [d for d in row if _has_letters(d.text)]
        pool = named if named else [anchor]
        pool.sort(key=lambda d: d.x)
        if len(pool) == 1:
            text = pool[0].text
        else:
            # Merge fragments left-to-right (handles split names like
            # "Sparkyy TTV [LIVE]"), but if the anchor outscores the rest it is
            # the clean read and the others are avatar noise in the same row.
            anchor_in = next((d for d in pool if d is anchor), pool[0])
            others = [d for d in pool if d is not anchor_in]
            if others and _score(anchor_in) >= 1.5 * (sum(_score(d) for d in others) / len(others)):
                text = anchor_in.text
            else:
                text = " ".join(d.text for d in pool)
        text = text.strip()
        if text:
            names.append(text)
    return names


def build_reader():
    import easyocr
    return easyocr.Reader(["en", "ru"], gpu=False, verbose=False)


def process(reader, image_path: str, tmpdir: str, dual: bool = True) -> list[str]:
    """Full pipeline for one image -> list of name strings.

    With ``dual=True`` (default, quality mode) two OCR passes are merged so each
    row keeps its best read:
      * Pass A — ffmpeg crop+upscale only.
      * Pass B — same + magick contrast stretch (recovers faint names).
    Clustering + dedup then naturally picks the higher-confidence variant per
    row. With ``dual=False`` only pass B runs (~2x faster).
    """
    scaled = os.path.join(tmpdir, "scaled.png")
    enhanced = os.path.join(tmpdir, "enhanced.png")
    ffmpeg_crop_scale(image_path, scaled)
    magick_enhance(scaled, enhanced)
    if dual:
        dets = ocr_image(reader, scaled) + ocr_image(reader, enhanced)
    else:
        dets = ocr_image(reader, enhanced)
    return extract_names(dets)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image paths to process")
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument("--test", action="store_true", help="accuracy report vs data/*.json")
    ap.add_argument("--fast", action="store_true",
                    help="single OCR pass (~2x faster, slightly less accurate)")
    args = ap.parse_args()

    if args.test:
        return run_test(args)

    if not args.images:
        ap.error("provide image paths or use --test")

    reader = build_reader()
    with tempfile.TemporaryDirectory() as tmpdir:
        for img in args.images:
            names = process(reader, img, tmpdir, dual=not args.fast)
            if args.json:
                print(json.dumps({"image": img, "names": names}))
            else:
                for n in names:
                    print(n)


def run_test(args):
    """Process every data/N.jpg and compare against data/N.json."""
    data = Path(__file__).parent.parent / "testdata"
    pairs = []
    for jf in sorted(data.glob("*.json"), key=lambda p: int(p.stem)):
        img = jf.with_suffix(".jpg")
        if img.exists():
            pairs.append((img, jf))

    reader = build_reader()
    total, matched = 0, 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for img, jf in pairs:
            expected = json.loads(jf.read_text())["survivors"]
            got = process(reader, str(img), tmpdir)
            print(f"### {img.name}")
            for exp in expected:
                total += 1
                ok = _best_match(exp, got)
                matched += ok
                mark = "OK " if ok else "   "
                print(f"  [{mark}] exp={exp!r}")
            print(f"       got={got}")
    print(f"\n=== {matched}/{total} names matched ===")


def _edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _best_match(expected: str, candidates: list[str]) -> int:
    """1 if some candidate is within edit distance of expected (tolerant)."""
    exp = expected.lower().replace(" ", "").rstrip(".")
    tol = max(2, len(exp) // 5)
    for c in candidates:
        got = c.lower().replace(" ", "")
        if _edit_distance(exp, got) <= tol:
            return 1
    return 0


if __name__ == "__main__":
    main()
