import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from silero_vad import get_speech_timestamps, load_silero_vad

ASR_MODEL_NAME     = "Qwen/Qwen3-ASR-1.7B"
TARGET_SAMPLE_RATE = 16000

VAD_THRESHOLD           = 0.20
MIN_SPEECH_DURATION_MS  = 300
MIN_SILENCE_DURATION_MS = 600
SPEECH_PADDING_MS       = 100

MAX_REGIONS     = 20
MAX_REGION_SECS = 30.0
BATCH_SIZE      = 4


def load_audio(path: str):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path, "-vn", "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if not result.stdout:
        raise RuntimeError("FFmpeg returned no audio.")

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    audio = np.nan_to_num(audio)
    if len(audio):
        audio -= np.mean(audio)
    return audio, TARGET_SAMPLE_RATE


def detect_speech_regions(audio, sample_rate) -> list[dict]:
    print("[ASR] Running VAD...", flush=True)
    vad = load_silero_vad()
    waveform = torch.from_numpy(audio)

    timestamps = get_speech_timestamps(
        waveform, vad,
        sampling_rate=sample_rate,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PADDING_MS,
        return_seconds=False,
    )

    regions = [
        {"start": int(t["start"]) / sample_rate, "end": int(t["end"]) / sample_rate}
        for t in timestamps if int(t["end"]) > int(t["start"])
    ]
    print(f"[ASR] VAD regions: {len(regions)}", flush=True)

    del vad, waveform
    gc.collect()
    return regions


def sample_regions(regions: list[dict]) -> list[dict]:
    if not regions:
        return []

    if len(regions) <= MAX_REGIONS:
        selected = regions
    else:
        step = len(regions) / MAX_REGIONS
        selected = [regions[int(i * step)] for i in range(MAX_REGIONS)]

    capped = []
    for r in selected:
        end = min(r["end"], r["start"] + MAX_REGION_SECS)
        if end - r["start"] >= 0.3:
            capped.append({"start": r["start"], "end": end})

    print(f"[ASR] Sampled {len(capped)} regions (max {MAX_REGION_SECS}s each)", flush=True)
    return capped


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    if torch.cuda.is_available():
        print(f"[ASR] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("[ASR] Loading Qwen3-ASR...", flush=True)
    t0 = time.time()

    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_NAME,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=BATCH_SIZE,
        max_new_tokens=256,
    )
    print(f"[ASR] Model ready in {time.time() - t0:.1f}s", flush=True)
    return model


def get_segment(audio, sr, start, end):
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    return audio[a:b] if b > a else np.empty(0, dtype=np.float32)


def transcribe_all(model, audio, sample_rate, regions, language="auto") -> str:
    parts = []
    total = len(regions)

    for i in range(0, total, BATCH_SIZE):
        batch_regions = regions[i:i + BATCH_SIZE]
        batch_audio = []

        for r in batch_regions:
            seg = get_segment(audio, sample_rate, r["start"], r["end"])
            if len(seg) / sample_rate >= 0.3:
                batch_audio.append((seg, sample_rate))

        if not batch_audio:
            continue

        idx_end = min(i + BATCH_SIZE, total)
        print(f"[ASR] Batch {i // BATCH_SIZE + 1}  regions {i+1}-{idx_end}/{total}", flush=True)

        try:
            kwargs = {} if language == "auto" else {"language": language}

            if len(batch_audio) == 1:
                results = model.transcribe(audio=batch_audio[0], **kwargs)
                results = [results] if not isinstance(results, list) else results
            else:
                results = model.transcribe(audio=batch_audio, **kwargs)
                if not isinstance(results, list):
                    results = [results]

            for result in results:
                items = result if isinstance(result, list) else [result]
                for item in items:
                    text = ""
                    if hasattr(item, "text") and item.text:
                        text = item.text
                    elif hasattr(item, "transcript") and item.transcript:
                        text = item.transcript
                    text = " ".join(str(text).split()).strip()
                    if text:
                        parts.append(text)

        except Exception as exc:
            print(f"[ASR] Batch error: {exc}", flush=True)
            continue

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    print("ASR WORKER", flush=True)

    audio, sr = load_audio(args.audio)
    duration  = len(audio) / sr
    print(f"[ASR] Duration: {duration:.0f}s ({duration/60:.1f}min)", flush=True)

    all_regions     = detect_speech_regions(audio, sr)
    sampled_regions = sample_regions(all_regions)

    model = load_model()
    text  = transcribe_all(model, audio, sr, sampled_regions, args.language)

    output = {
        "duration": round(duration, 2),
        "text":     text,
        "words":    len(text.split()) if text else 0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[ASR] Done. {output['words']} words -> {args.output}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[ASR] GPU memory released.", flush=True)


if __name__ == "__main__":
    main()
