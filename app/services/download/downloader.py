import asyncio
from pathlib import Path
from datetime import datetime
import subprocess
import sys

from app.core.config import settings
from app.core.logging import get_logger
from app.core.enums import Platform

logger = get_logger(__name__)


class DownloadError(Exception):
    pass


class VideoDownloader:
    def __init__(self):
        self.storage_path = Path(settings.media_storage_path)
        self.candidates_folder = self.storage_path / settings.candidates_folder
        self.candidates_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"VideoDownloader ready - candidates: {self.candidates_folder}")

    async def download_candidate(self, url: str, video_id: str, title: str) -> Path:
        """Download video at highest quality and merge to MP4"""
        logger.info(f"Downloading: {title}")

        try:
            path = await self._download(url, video_id, title)
            logger.info(f"Downloaded to: {path}")
            return path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise DownloadError(f"Download failed: {e}")

    async def _download(self, url: str, video_id: str, title: str) -> Path:
        """Use yt-dlp to download highest quality video"""
        
        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title[:50]  # Limit length
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_id}_{timestamp}.mp4"
        output_path = self.candidates_folder / filename

        # Format selector: best video + best audio, merge to mp4
        format_selector = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f", format_selector,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--progress",
            url
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace').strip()
                logger.error(f"yt-dlp error: {error_msg}")
                raise DownloadError(f"yt-dlp failed: {error_msg}")

            if not output_path.exists():
                raise DownloadError(f"File not found after download: {output_path}")

            return output_path

        except FileNotFoundError:
            raise DownloadError("yt-dlp not found. Install with: pip install yt-dlp")

    def get_candidates_folder(self) -> Path:
        return self.candidates_folder

    def list_candidates(self) -> list[Path]:
        return list(self.candidates_folder.glob("*.mp4"))
