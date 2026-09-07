import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.discovery.youtube import (
    YouTubeDiscoveryProvider,
    YouTubeAPIAuthError,
    YouTubeAPIRateLimitError,
    YouTubeAPIRequestError,
    YouTubeAPITimeoutError,
    YouTubeAPIError
)

from app.services.download.downloader import VideoDownloader
from app.services.download.queue import DownloadQueue, DownloadStatus
from app.core.enums import (
    VideoDuration, SearchOrder, VideoType, SafeSearch,
    VideoDefinition, VideoCaption, VideoLicense, EventType, VideoDimension
)

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
youtube_provider = None
video_downloader = None
download_queue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global youtube_provider, video_downloader, download_queue
    try:
        logger.info("Starting NABLET API...")
        youtube_provider = YouTubeDiscoveryProvider()
        video_downloader = VideoDownloader()
        download_queue = DownloadQueue(video_downloader, max_concurrent=3)
        await download_queue.start()
        logger.info("NABLET API ready on http://localhost:8000")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    if download_queue:
        await download_queue.stop()
    if youtube_provider:
        await youtube_provider.close()


# FastAPI app
app = FastAPI(title="NABLET API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === REQUEST/RESPONSE MODELS ===

class SearchRequest(BaseModel):
    keywords: Optional[List[str]] = None
    phrases: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    include_terms: Optional[List[str]] = None
    exclude_terms: Optional[List[str]] = None
    published_after_days: Optional[int] = None
    duration: str = "any"
    order: str = "date"
    language: Optional[str] = None
    region: Optional[str] = None
    max_results: int = Field(10, ge=1, le=200)
    safe_search: str = "moderate"
    video_definition: Optional[str] = None
    video_dimension: Optional[str] = None
    video_caption: Optional[str] = None
    video_license: Optional[str] = None
    event_type: Optional[str] = None
    channel_ids: Optional[List[str]] = None
    video_category_id: Optional[str] = None


class VideoResponse(BaseModel):
    platform: str
    platform_video_id: str
    url: str
    title: str
    description: Optional[str]
    channel_name: str
    channel_url: Optional[str] = None
    thumbnail_url: Optional[str]
    duration_seconds: Optional[int]
    view_count: Optional[int] = None
    published_at: Optional[datetime] = None


class SearchResponse(BaseModel):
    success: bool
    count: int
    videos: List[VideoResponse]
    message: Optional[str] = None


class DownloadRequest(BaseModel):
    videos: List[dict]


class DownloadTaskResponse(BaseModel):
    video_id: str
    title: str
    status: str
    filename: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class QueueStatusResponse(BaseModel):
    queue_size: int
    total_tasks: int
    pending: int
    downloading: int
    completed: int
    failed: int
    tasks: List[DownloadTaskResponse]


# === ENUM VALIDATION HELPERS ===

def validate_enum(value: str, enum_map: dict, field_name: str):
    """Validate and convert string to enum"""
    if not value:
        return None
    value_lower = value.lower()
    if value_lower not in enum_map:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: {value}. Must be one of: {', '.join(enum_map.keys())}"
        )
    return enum_map[value_lower]


# === ENDPOINTS ===

@app.get("/")
async def root():
    """API root endpoint"""
    return {"service": "NABLET API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "youtube_api": "connected" if youtube_provider else "not initialized",
        "download_queue": "running" if download_queue and download_queue.running else "stopped"
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """Search YouTube videos with filters"""
    logger.info(f"Search request: {request.model_dump()}")

    try:
        # Date filter
        published_after = None
        if request.published_after_days:
            published_after = datetime.now(timezone.utc) - timedelta(days=request.published_after_days)

        # Validate required enums
        duration = validate_enum(request.duration, {
            'short': VideoDuration.SHORT,
            'medium': VideoDuration.MEDIUM,
            'long': VideoDuration.LONG,
            'any': VideoDuration.ANY
        }, 'duration') or VideoDuration.ANY

        order = validate_enum(request.order, {
            'date': SearchOrder.DATE,
            'relevance': SearchOrder.RELEVANCE,
            'viewcount': SearchOrder.VIEW_COUNT,
            'rating': SearchOrder.RATING,
            'title': SearchOrder.TITLE
        }, 'order') or SearchOrder.DATE

        safe_search = validate_enum(request.safe_search, {
            'none': SafeSearch.NONE,
            'moderate': SafeSearch.MODERATE,
            'strict': SafeSearch.STRICT
        }, 'safe_search') or SafeSearch.MODERATE

        # Validate optional enums
        video_definition = validate_enum(request.video_definition, {
            'any': VideoDefinition.ANY,
            'high': VideoDefinition.HIGH,
            'standard': VideoDefinition.STANDARD
        }, 'video_definition')

        video_dimension = validate_enum(request.video_dimension, {
            'any': VideoDimension.ANY,
            '2d': VideoDimension.TWO_D,
            '3d': VideoDimension.THREE_D
        }, 'video_dimension')

        video_caption = validate_enum(request.video_caption, {
            'any': VideoCaption.ANY,
            'closedcaption': VideoCaption.CLOSED_CAPTION,
            'none': VideoCaption.NONE
        }, 'video_caption')

        video_license = validate_enum(request.video_license, {
            'any': VideoLicense.ANY,
            'creativecommon': VideoLicense.CREATIVE_COMMON,
            'youtube': VideoLicense.YOUTUBE
        }, 'video_license')

        event_type = validate_enum(request.event_type, {
            'any': EventType.ANY,
            'completed': EventType.COMPLETED,
            'live': EventType.LIVE,
            'upcoming': EventType.UPCOMING
        }, 'event_type')

        # Execute search
        results = await youtube_provider.search(
            keywords=request.keywords,
            phrases=request.phrases,
            hashtags=request.hashtags,
            include_terms=request.include_terms,
            exclude_terms=request.exclude_terms,
            published_after=published_after,
            duration=duration,
            order=order,
            language=request.language,
            region=request.region,
            max_results=request.max_results,
            video_type=VideoType.VIDEO,  # Always video for /api/search
            safe_search=safe_search,
            video_definition=video_definition,
            video_dimension=video_dimension,
            video_caption=video_caption,
            video_license=video_license,
            event_type=event_type,
            channel_ids=request.channel_ids,
            video_category_id=request.video_category_id,
        )

        videos = [VideoResponse(**video) for video in results]
        logger.info(f"Search returned {len(videos)} results")

        return SearchResponse(
            success=True,
            count=len(videos),
            videos=videos,
            message=f"Found {len(videos)} videos"
        )

    except YouTubeAPIAuthError as e:
        logger.error(f"YouTube auth error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or unauthorized YouTube API key")

    except YouTubeAPIRateLimitError as e:
        logger.error(f"YouTube rate limit: {e}")
        raise HTTPException(status_code=429, detail="YouTube API quota exceeded. Please try again later.")

    except YouTubeAPIRequestError as e:
        logger.error(f"YouTube request error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid search parameters: {e}")

    except YouTubeAPITimeoutError as e:
        logger.error(f"YouTube timeout: {e}")
        raise HTTPException(status_code=504, detail="YouTube API request timed out")

    except YouTubeAPIError as e:
        logger.error(f"YouTube API error: {e}")
        raise HTTPException(status_code=502, detail=f"YouTube API error: {e}")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected search error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/download/add")
async def add_to_download_queue(request: DownloadRequest):
    """Add videos to download queue"""
    try:
        tasks = await download_queue.add_batch(request.videos)
        logger.info(f"Added {len(tasks)} videos to download queue")
        return {
            "success": True,
            "added": len(tasks),
            "message": f"Added {len(tasks)} videos to download queue"
        }
    except Exception as e:
        logger.error(f"Add to queue error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/queue", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get download queue status"""
    try:
        tasks = download_queue.get_all_tasks()

        stats = {
            'pending': sum(1 for t in tasks if t.status == DownloadStatus.PENDING),
            'downloading': sum(1 for t in tasks if t.status == DownloadStatus.DOWNLOADING),
            'completed': sum(1 for t in tasks if t.status == DownloadStatus.COMPLETED),
            'failed': sum(1 for t in tasks if t.status == DownloadStatus.FAILED),
        }

        task_responses = [
            DownloadTaskResponse(
                video_id=t.video_id,
                title=t.title,
                status=t.status.value,
                filename=t.filename,
                error=t.error,
                created_at=t.created_at,
                completed_at=t.completed_at
            )
            for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)
        ]

        return QueueStatusResponse(
            queue_size=download_queue.get_queue_size(),
            total_tasks=len(tasks),
            **stats,
            tasks=task_responses
        )
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/channels/search")
async def search_channels(q: str):
    """Search for YouTube channels"""
    if not q or len(q) < 2:
        return {"channels": []}

    try:
        results = await youtube_provider.search(
            keywords=[q],
            video_type=VideoType.CHANNEL,
            max_results=10
        )

        channels = [{
            "id": r.get("platform_video_id") or r.get("channel_id"),
            "title": r.get("channel_name") or r.get("title"),
            "thumbnail": f"/api/proxy/image?url={r.get('thumbnail_url')}" if r.get('thumbnail_url') else None,
            "description": (r.get("description") or "")[:100]
        } for r in results]

        logger.info(f"Channel search '{q}': {len(channels)} results")
        return {"channels": channels}

    except YouTubeAPIAuthError as e:
        logger.error(f"YouTube auth error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or unauthorized YouTube API key")

    except YouTubeAPIRateLimitError as e:
        logger.error(f"YouTube rate limit: {e}")
        raise HTTPException(status_code=429, detail="YouTube API quota exceeded")

    except YouTubeAPITimeoutError as e:
        logger.error(f"YouTube timeout: {e}")
        raise HTTPException(status_code=504, detail="YouTube API request timed out")

    except YouTubeAPIError as e:
        logger.error(f"YouTube API error: {e}")
        raise HTTPException(status_code=502, detail=f"YouTube API error: {e}")

    except Exception as e:
        logger.error(f"Channel search error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/proxy/image")
async def proxy_image(url: str):
    """Proxy YouTube images to avoid CORS issues"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "image/jpeg"),
                headers={"Cache-Control": "public, max-age=86400"}
            )
    except Exception as e:
        logger.error(f"Image proxy error: {e}")
        raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
