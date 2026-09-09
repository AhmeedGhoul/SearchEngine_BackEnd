"""
OCR Worker   On-screen text extraction for keyword extraction.

Runs as an isolated subprocess under .venv_ocr (PaddlePaddle CUDA).
Samples frames uniformly then filters duplicates   simple and fast
because we only need keyword evidence, not precise timestamps.

Usage:
    python ocr_worker.py --video <path> --output <json_path>
                         [--sample-every 5.0]
                         [--adaptive-schedule <json_path>]
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


# ── CONFIG ────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_EVERY = 10.0   # seconds between frame checks
MIN_TEXT_SCORE       = 0.50
MIN_ALNUM_CHARS      = 6      # slightly lower than highlight pipeline

_RE_WS = re.compile(r"\s+")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return _RE_WS.sub(" ", str(text)).strip() if text else ""


def is_noise(text: str) -> bool:
    if not text:
        return True
    alnum = sum(1 for c in text if c.isalnum())
    if alnum < MIN_ALNUM_CHARS:
        return True
    # pure punctuation / symbols
    if all(
        unicodedata.category(c).startswith("P")
        or unicodedata.category(c).startswith("S")
        or c.isspace()
        for c in text
    ):
        return True
    return False


def token_overlap(a: str, b: str) -> float:
    ta = set(_RE_WS.sub(" ", a.lower()).split())
    tb = set(_RE_WS.sub(" ", b.lower()).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_text(result) -> str:
    try:
        raw = result.json
        if isinstance(raw, str):
            raw = json.loads(raw)
    except Exception:
        return ""

    if not isinstance(raw, dict):
        return ""

    res = raw.get("res", raw)
    if not isinstance(res, dict):
        return ""

    rec_texts  = res.get("rec_texts",  [])
    rec_scores = res.get("rec_scores", [])

    texts = []
    for i, text in enumerate(rec_texts):
        text = normalize(text)
        if not text:
            continue
        if i < len(rec_scores):
            try:
                if float(rec_scores[i]) < MIN_TEXT_SCORE:
                    continue
            except (TypeError, ValueError):
                pass
        texts.append(text)

    return normalize(" ".join(texts))


def run_ocr(ocr, frame) -> str:
    try:
        prediction = list(ocr.predict(frame))
    except Exception as exc:
        print(f"[OCR] Prediction error: {exc}", flush=True)
        return ""

    texts = [extract_text(r) for r in prediction]
    return normalize(" ".join(t for t in texts if t))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--sample-every", type=float, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--adaptive-schedule", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("OCR WORKER   KEYWORD EXTRACTION MODE", flush=True)
    print("=" * 60, flush=True)

    import paddle
    print(f"[OCR] PaddlePaddle {paddle.__version__} | CUDA: {paddle.device.is_compiled_with_cuda()}", flush=True)

    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError("OCR environment requires a CUDA PaddlePaddle build.")

    print("[OCR] Loading PP-OCRv6...", flush=True)
    t0  = time.time()
    ocr = PaddleOCR(
        lang="en",
        device="gpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print(f"[OCR] Model ready in {time.time() - t0:.1f}s", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = frame_count / fps if fps > 0 else 0

    # Build sample list
    sample_times = None
    if args.adaptive_schedule:
        p = Path(args.adaptive_schedule)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sample_times = data.get("sample_times", [])
                print(f"[OCR] Adaptive schedule: {len(sample_times)} samples", flush=True)
            except Exception as e:
                print(f"[OCR] Schedule load failed: {e}   using uniform sampling", flush=True)

    if sample_times is None:
        interval = max(0.5, args.sample_every)
        sample_times = [round(t, 2) for t in
                        (i * interval for i in range(int(duration / interval) + 1))
                        if t <= duration]
        print(f"[OCR] Uniform sampling: {len(sample_times)} frames every {interval}s", flush=True)

    target_frames = [min(frame_count - 1, int(t * fps)) for t in sample_times]
    total         = len(target_frames)

    results        = []
    seen_texts     = []   # (text, timestamp)   for dedup
    DEDUP_WINDOW   = 30.0  # seconds
    DEDUP_OVERLAP  = 0.70  # token overlap threshold

    for idx, frame_no in enumerate(target_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
        ret, frame = cap.read()
        if not ret:
            continue

        ts   = frame_no / fps
        text = run_ocr(ocr, frame)

        if not text or is_noise(text):
            print(f"[OCR] {idx}/{total}  {ts:.1f}s  (noise/empty)", flush=True)
            continue

        # Dedup: skip if very similar text was seen recently
        is_dup = False
        for (prev_text, prev_ts) in seen_texts:
            if abs(ts - prev_ts) < DEDUP_WINDOW and token_overlap(text, prev_text) >= DEDUP_OVERLAP:
                is_dup = True
                break

        if is_dup:
            print(f"[OCR] {idx}/{total}  {ts:.1f}s  (duplicate)", flush=True)
            continue

        print(f"[OCR] {idx}/{total}  {ts:.1f}s  → {text[:80]}", flush=True)
        results.append({"time": round(ts, 2), "text": text})
        seen_texts.append((text, ts))

        # Keep only recent history
        seen_texts = [(t, ts_) for (t, ts_) in seen_texts
                      if ts - ts_ <= DEDUP_WINDOW]

    cap.release()

    output = {"duration": round(duration, 2), "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OCR] Done. {len(results)} unique text frames → {args.output}", flush=True)


if __name__ == "__main__":
    main()
