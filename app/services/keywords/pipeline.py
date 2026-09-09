import asyncio
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL     = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 300

LLM_MODEL = "qwen3:14b"
VLM_MODEL = "qwen3-VL:4B"

ROOT     = Path(__file__).resolve().parents[3]
VENV_ASR = ROOT / ".venv_asr" / "Scripts" / "python.exe"

WORKERS_DIR    = Path(__file__).resolve().parent
ASR_WORKER     = WORKERS_DIR / "asr_worker.py"
VLM_WORKER     = WORKERS_DIR / "vlm_worker.py"
WORKER_TIMEOUT = 3600


def _run_worker(python_exe: Path, script: Path, args: list[str], log_path: Path) -> bool:
    cmd = [str(python_exe), str(script)] + args
    logger.info(f"Launching: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                logger.debug(line.rstrip())
            proc.wait(timeout=WORKER_TIMEOUT)

        if proc.returncode != 0:
            logger.error(f"Worker exited {proc.returncode}   see {log_path}")
            return False
        return True

    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error(f"Worker timed out after {WORKER_TIMEOUT}s")
        return False
    except Exception as exc:
        logger.error(f"Worker error: {exc}")
        return False


async def _run_worker_async(python_exe: Path, script: Path, args: list[str], log_path: Path) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_worker, python_exe, script, args, log_path)


_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_keyword_prompt(asr_text: str, vlm_obs: list[dict], duration: float) -> str:
    duration_min = duration / 60.0
    vlm_block = "\n".join(o["text"] for o in vlm_obs if o.get("text"))

    return f"""You are a YouTube search strategist. Analyze this video and produce search parameters to find OTHER SIMILAR videos   not this exact one.

Goal: what would someone search on YouTube to find videos on the same topic, style, or theme?

Video duration: {duration_min:.0f} minutes

SPEECH:
{asr_text.strip()[:6000] if asr_text.strip() else "(none)"}

VISUALS:
{vlm_block.strip()[:2000] if vlm_block.strip() else "(none)"}

HOW YOUTUBE SEARCH WORKS:

keywords   main search terms (YouTube matches titles, descriptions, tags)
  - 2-5 words each
  - good: "food industry lobbying", "ultra processed food documentary"
  - bad: too specific like "Nestle Switzerland 2023" or too vague like "food"

phrases   exact phrase YouTube must find in title/description
  - only use if the video has a very distinctive phrase
  - leave empty if nothing stands out

hashtags   topic tags creators use (without #)
  - example: fooddocumentary, obesity, nutrition

include_terms   words ALL results must contain (use to narrow down)

exclude_terms   words to filter out (use to remove irrelevant results)

BALANCE:
- not too specific (no exact names, unique events, or this video's title)
- not too general ("food" returns millions of unrelated videos)
- aim for: same theme or debate from different angles

Return ONLY valid JSON:
{{
  "keywords": [
    {{"kw": "search term", "category": "topic|entity|action|style"}},
    ...
  ],
  "phrases": [],
  "hashtags": [],
  "include_terms": [],
  "exclude_terms": []
}}

keywords: 10-20 terms, most important first
phrases: 0-3, only if clearly distinctive
hashtags: 0-5
include_terms: 0-3
exclude_terms: 0-3"""


def parse_llm_response(text: str) -> dict:
    text = _RE_THINK.sub("", text.strip()).strip()

    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    def extract(raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    data = extract(text)
    return {
        "keywords":      data.get("keywords", []),
        "phrases":       data.get("phrases", []),
        "hashtags":      data.get("hashtags", []),
        "include_terms": data.get("include_terms", []),
        "exclude_terms": data.get("exclude_terms", []),
    }


async def call_llm(prompt: str) -> dict:
    payload = {
        "model":    LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "options":  {"temperature": 0.3, "num_predict": 2048},
        "think":    False,
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return parse_llm_response(content)


class PipelineStatus:
    def __init__(self):
        self.steps:      list[dict]  = []
        self.current:    str         = ""
        self.done:       bool        = False
        self.error:      str | None  = None
        self.result:     dict | None = None
        self._callbacks: list        = []

    def update(self, step: str, message: str = ""):
        self.current = step
        entry = {"step": step, "message": message}
        self.steps.append(entry)
        logger.info(f"[PIPELINE] {step}: {message}")
        for cb in self._callbacks:
            cb(entry)

    def finish(self, result: dict):
        self.result = result
        self.done = True
        for cb in self._callbacks:
            cb({"step": "done", "result": result})

    def fail(self, error: str):
        self.error = error
        self.done = True
        for cb in self._callbacks:
            cb({"step": "error", "error": error})


async def run_keyword_pipeline(
    video_path:     str,
    processing_dir: str,
    run_asr:        bool = True,
    run_vlm:        bool = True,
    language:       str  = "auto",
    status:         Optional[PipelineStatus] = None,
) -> dict:
    if status is None:
        status = PipelineStatus()

    proc_dir = Path(processing_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    audio_path = proc_dir / "audio.wav"
    asr_json   = proc_dir / "asr_results.json"
    vlm_json   = proc_dir / "vlm_results.json"
    asr_log    = proc_dir / "asr_worker.log"
    vlm_log    = proc_dir / "vlm_worker.log"

    video = Path(video_path)
    if not video.exists():
        status.fail(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = 0.0
    asr_text = ""
    vlm_data = {"observations": []}

    # extract audio
    status.update("audio", "Extracting audio track...")
    try:
        import subprocess as sp
        result = sp.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(audio_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 or not audio_path.exists():
            raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")

        dur_result = sp.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=False,
        )
        try:
            duration = float(dur_result.stdout.strip())
        except ValueError:
            duration = 0.0

        status.update("audio", f"Audio extracted ({duration:.0f}s)")
    except Exception as exc:
        status.fail(f"Audio extraction failed: {exc}")
        raise

    # speech recognition
    if run_asr:
        status.update("asr", "Running speech recognition...")
        if not VENV_ASR.exists():
            logger.warning(f".venv_asr not found at {VENV_ASR}   skipping ASR")
            status.update("asr", "Skipped (.venv_asr not found)")
        else:
            ok = await _run_worker_async(
                VENV_ASR, ASR_WORKER,
                ["--audio", str(audio_path), "--output", str(asr_json), "--language", language],
                asr_log,
            )
            if ok and asr_json.exists():
                try:
                    data     = json.loads(asr_json.read_text(encoding="utf-8"))
                    asr_text = data.get("text", "").strip()
                    words    = data.get("words", len(asr_text.split()))
                    status.update("asr", f"Transcribed {words} words")
                except Exception as e:
                    logger.warning(f"ASR JSON parse error: {e}")
                    status.update("asr", "ASR completed (parse warning)")
            else:
                status.update("asr", "ASR failed   continuing without speech")

    # visual analysis
    if run_vlm:
        status.update("vlm", "Running visual analysis...")
        ok = await _run_worker_async(
            Path(sys.executable), VLM_WORKER,
            ["--video", str(video), "--output", str(vlm_json),
             "--sample-every", "10", "--model", VLM_MODEL, "--ollama-url", OLLAMA_URL],
            vlm_log,
        )
        if ok and vlm_json.exists():
            try:
                vlm_data  = json.loads(vlm_json.read_text(encoding="utf-8"))
                n_batches = vlm_data.get("batch_count", 0)
                n_frames  = vlm_data.get("frame_count", 0)
                status.update("vlm", f"Analyzed {n_frames} frames in {n_batches} batches")
            except Exception as e:
                logger.warning(f"VLM JSON parse error: {e}")
                status.update("vlm", "VLM completed (parse warning)")
        else:
            status.update("vlm", "VLM failed or Ollama not running   continuing")

    # generate keywords
    status.update("llm", "Generating keywords with LLM...")
    vlm_obs = [o for o in vlm_data.get("observations", []) if o.get("text", "").strip()]
    try:
        prompt        = build_keyword_prompt(asr_text, vlm_obs, duration)
        search_params = await call_llm(prompt)
        keywords      = search_params.get("keywords", [])
        status.update("llm", f"Generated {len(keywords)} keywords")
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        status.update("llm", f"LLM failed: {exc}")
        search_params = {"keywords": [], "phrases": [], "hashtags": [],
                         "include_terms": [], "exclude_terms": []}
        keywords = []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(f"{OLLAMA_URL}/api/generate",
                              json={"model": LLM_MODEL, "keep_alive": 0})
        logger.info("LLM model unloaded from VRAM")
    except Exception:
        pass

    result = {
        "keywords":         keywords,
        "phrases":          search_params.get("phrases", []),
        "hashtags":         search_params.get("hashtags", []),
        "include_terms":    search_params.get("include_terms", []),
        "exclude_terms":    search_params.get("exclude_terms", []),
        "asr_word_count":   len(asr_text.split()) if asr_text else 0,
        "vlm_batch_count":  vlm_data.get("batch_count", 0),
        "duration_seconds": duration,
    }
    status.finish(result)
    return result
