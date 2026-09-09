import argparse
import base64
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import httpx

DEFAULT_SAMPLE_EVERY = 10
DEFAULT_MODEL        = "qwen3-VL:4B"
DEFAULT_OLLAMA       = "http://127.0.0.1:11434"

MAX_FRAME_DIM    = 320
OLLAMA_TIMEOUT   = 300
FRAMES_PER_BATCH = 5
MIN_BATCHES      = 2
MAX_BATCHES      = 4
DIVERSITY_LOW    = 8
DIVERSITY_HIGH   = 22


def phash(frame: np.ndarray, size: int = 8) -> int:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size * 4, size * 4), interpolation=cv2.INTER_AREA)
    dct   = cv2.dct(small.astype(np.float32))
    top   = dct[:size, :size]
    bits  = (top > top.mean()).flatten()
    val   = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def mean_diversity(hashes: list[int]) -> float:
    if len(hashes) < 2:
        return 0.0
    total = 0
    count = 0
    step = max(1, len(hashes) // 15)
    for i in range(0, len(hashes), step):
        for j in range(i + 1, min(i + step + 1, len(hashes))):
            total += hamming(hashes[i], hashes[j])
            count += 1
    return total / count if count else 0.0


def sample_distinct_frames(video_path: str, sample_every: float) -> tuple[list[dict], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = frame_count / fps if fps > 0 else 0.0

    print(f"[VLM] Duration: {duration:.1f}s  FPS: {fps:.1f}", flush=True)

    interval     = max(1.0, sample_every)
    sample_times = [i * interval for i in range(int(duration / interval) + 1)
                    if i * interval <= duration]

    raw = []
    for ts in sample_times:
        fn = min(frame_count - 1, int(ts * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fn))
        ret, frame = cap.read()
        if ret:
            raw.append({"timestamp": round(ts, 1), "frame": frame, "hash": phash(frame)})

    cap.release()
    print(f"[VLM] Raw samples: {len(raw)}", flush=True)

    if not raw:
        return [], duration

    kept      = [raw[0]]
    last_hash = raw[0]["hash"]
    for f in raw[1:]:
        if hamming(f["hash"], last_hash) >= 3:
            kept.append(f)
            last_hash = f["hash"]

    print(f"[VLM] Distinct frames: {len(kept)}", flush=True)
    return kept, duration


def decide_batch_count(frames: list[dict]) -> int:
    if not frames:
        return MIN_BATCHES

    hashes    = [f["hash"] for f in frames]
    diversity = mean_diversity(hashes)
    print(f"[VLM] Visual diversity: {diversity:.2f}", flush=True)

    if diversity <= DIVERSITY_LOW:
        batches = MIN_BATCHES
        reason  = "low diversity"
    elif diversity >= DIVERSITY_HIGH:
        batches = MAX_BATCHES
        reason  = "high diversity"
    else:
        ratio   = (diversity - DIVERSITY_LOW) / (DIVERSITY_HIGH - DIVERSITY_LOW)
        batches = round(MIN_BATCHES + ratio * (MAX_BATCHES - MIN_BATCHES))
        reason  = "medium diversity"

    max_possible = max(1, len(frames) // FRAMES_PER_BATCH) if len(frames) >= FRAMES_PER_BATCH else 1
    batches      = min(batches, max_possible, MAX_BATCHES)
    batches      = max(batches, min(MIN_BATCHES, max_possible))

    print(f"[VLM] Batches: {batches} ({reason})", flush=True)
    return batches


def spread_frames(frames: list[dict], n_batches: int) -> list[list[dict]]:
    if not frames or n_batches <= 0:
        return []

    target_total = min(n_batches * FRAMES_PER_BATCH, len(frames))
    indices  = [round(i * (len(frames) - 1) / (target_total - 1))
                for i in range(target_total)] if target_total > 1 else [0]
    selected = [frames[i] for i in sorted(set(indices))]

    batches = []
    for i in range(0, len(selected), FRAMES_PER_BATCH):
        chunk = selected[i:i + FRAMES_PER_BATCH]
        if chunk:
            batches.append(chunk)

    return batches


def encode_frame(frame: np.ndarray) -> str:
    h, w  = frame.shape[:2]
    scale = min(MAX_FRAME_DIM / w, MAX_FRAME_DIM / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def call_vlm_batch(batch: list[dict], model: str, ollama_url: str) -> str:
    prompt = (
        "Analyze these video frames. "
        "List what you see: objects, people, characters, text on screen, "
        "actions, setting, style (cartoon/live/animation/etc.). "
        "Be brief and specific. Use commas to separate items. "
        "No full sentences. Focus on things useful as YouTube search keywords."
    )

    payload = {
        "model":    model,
        "messages": [{
            "role":    "user",
            "content": prompt,
            "images":  [encode_frame(f["frame"]) for f in batch],
        }],
        "stream":  False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
            "num_ctx":     32768,
        },
    }

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        resp = client.post(f"{ollama_url}/api/chat", json=payload)
        if not resp.is_success:
            raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:500]}")

    return resp.json().get("message", {}).get("content", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--sample-every", type=float, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url",   default=DEFAULT_OLLAMA)
    args = parser.parse_args()

    print("VLM WORKER", flush=True)

    frames, duration = sample_distinct_frames(args.video, args.sample_every)
    if not frames:
        print("[VLM] No frames extracted.", flush=True)
        output = {"frame_count": 0, "batch_count": 0, "observations": [], "duration": 0}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f)
        return

    n_batches = decide_batch_count(frames)
    batches   = spread_frames(frames, n_batches)
    print(f"[VLM] Running {len(batches)} batches x {FRAMES_PER_BATCH} frames", flush=True)

    observations = []
    for idx, batch in enumerate(batches, 1):
        ts_start = batch[0]["timestamp"]
        ts_end   = batch[-1]["timestamp"]
        print(f"[VLM] Batch {idx}/{len(batches)} ({len(batch)} frames, {ts_start:.0f}s-{ts_end:.0f}s)", flush=True)
        try:
            t0   = time.time()
            text = call_vlm_batch(batch, args.model, args.ollama_url)
            print(f"[VLM] Batch {idx} done in {time.time() - t0:.1f}s", flush=True)
            if text.strip():
                observations.append({
                    "batch":      idx,
                    "time_start": ts_start,
                    "time_end":   ts_end,
                    "text":       text.strip(),
                })
        except Exception as exc:
            print(f"[VLM] Batch {idx} error: {exc}", flush=True)

    output = {
        "duration":     round(duration, 2),
        "frame_count":  len(frames),
        "batch_count":  len(batches),
        "observations": observations,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[VLM] Done. {len(observations)} observations -> {args.output}", flush=True)

    try:
        with httpx.Client(timeout=30) as client:
            client.post(f"{args.ollama_url}/api/generate",
                        json={"model": args.model, "keep_alive": 0})
        print("[VLM] Model unloaded.", flush=True)
    except Exception as e:
        print(f"[VLM] Could not unload model: {e}", flush=True)

    try:
        subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe"],
                       capture_output=True, check=False)
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                       capture_output=True, check=False)
        print("[VLM] Ollama stopped.", flush=True)
    except Exception as e:
        print(f"[VLM] Could not stop Ollama: {e}", flush=True)


if __name__ == "__main__":
    main()
