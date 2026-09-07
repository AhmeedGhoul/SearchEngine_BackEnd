import asyncio
from typing import Dict, List
from datetime import datetime
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


class DownloadTask:
    def __init__(self, video_id: str, url: str, title: str):
        self.video_id = video_id
        self.url = url
        self.title = title
        self.status = DownloadStatus.PENDING
        self.error: str | None = None
        self.filename: str | None = None
        self.created_at = datetime.utcnow()
        self.completed_at: datetime | None = None


class DownloadQueue:
    def __init__(self, downloader, max_concurrent=3):
        self.downloader = downloader
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, DownloadTask] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False

    async def start(self):
        """Start queue workers"""
        if self.running:
            return
        
        self.running = True
        logger.info(f"Starting download queue with {self.max_concurrent} workers")
        
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

    async def stop(self):
        """Stop queue workers"""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("Download queue stopped")

    async def add_task(self, video_id: str, url: str, title: str) -> DownloadTask:
        """Add a download task to queue"""
        if video_id in self.tasks:
            return self.tasks[video_id]
        
        task = DownloadTask(video_id, url, title)
        self.tasks[video_id] = task
        await self.queue.put(task)
        logger.info(f"Added to queue: {title}")
        return task

    async def add_batch(self, videos: List[Dict]) -> List[DownloadTask]:
        """Add multiple videos to queue"""
        tasks = []
        for video in videos:
            task = await self.add_task(
                video['platform_video_id'],
                video['url'],
                video['title']
            )
            tasks.append(task)
        return tasks

    async def _worker(self, worker_id: int):
        """Worker that processes download tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                logger.info(f"Worker {worker_id} processing: {task.title}")
                task.status = DownloadStatus.DOWNLOADING

                try:
                    path = await self.downloader.download_candidate(
                        task.url,
                        task.video_id,
                        task.title
                    )
                    task.status = DownloadStatus.COMPLETED
                    task.filename = path.name
                    task.completed_at = datetime.utcnow()
                    logger.info(f"Worker {worker_id} completed: {task.title}")

                except Exception as e:
                    task.status = DownloadStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow()
                    logger.error(f"Worker {worker_id} failed: {task.title} - {e}")

                finally:
                    self.queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} stopped")

    def get_task(self, video_id: str) -> DownloadTask | None:
        """Get task status"""
        return self.tasks.get(video_id)

    def get_all_tasks(self) -> List[DownloadTask]:
        """Get all tasks"""
        return list(self.tasks.values())

    def get_queue_size(self) -> int:
        """Get number of pending tasks"""
        return self.queue.qsize()
