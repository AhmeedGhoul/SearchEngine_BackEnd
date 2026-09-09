import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
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
from app.services.fingerprinting import NabletFingerprintClient
from app.services.keywords import run_keyword_pipeline, PipelineStatus
from app.core.enums import (
    VideoDuration, SearchOrder, VideoType, SafeSearch,
    VideoDefinition, VideoCaption, VideoLicense, EventType, VideoDimension
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

youtube_provider = None
video_downloader = None
download_queue = None
fingerprint_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global youtube_provider, video_downloader, download_queue, fingerprint_client
    try:
        logger.info("Starting NABLET API...")
        youtube_provider = YouTubeDiscoveryProvider()
        video_downloader = VideoDownloader()
        download_queue = DownloadQueue(video_downloader, max_concurrent=3)
        await download_queue.start()
        
        fingerprint_client = NabletFingerprintClient()
        logger.info("Fingerprint client initialized")
        
        logger.info("NABLET API ready on http://localhost:8000")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    if download_queue:
        await download_queue.stop()
    if youtube_provider:
        await youtube_provider.close()


app = FastAPI(title="NABLET API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def validate_enum(value: str, enum_map: dict, field_name: str):
    if not value:
        return None
    value_lower = value.lower()
    if value_lower not in enum_map:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: {value}. Must be one of: {', '.join(enum_map.keys())}"
        )
    return enum_map[value_lower]


@app.get("/")
async def root():
    return {"service": "NABLET API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "youtube_api": "connected" if youtube_provider else "not initialized",
        "download_queue": "running" if download_queue and download_queue.running else "stopped"
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    logger.info(f"Search request: {request.model_dump()}")

    try:
        published_after = None
        if request.published_after_days:
            published_after = datetime.now(timezone.utc) - timedelta(days=request.published_after_days)

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
            video_type=VideoType.VIDEO,
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


@app.post("/api/fingerprint/generate")
async def generate_fingerprints():
    try:
        candidates_path = Path("media_storage/candidates")
        fpraw_output = Path("media_storage/fpraw")
        
        if not candidates_path.exists():
            raise HTTPException(status_code=404, detail="Candidates folder not found")
        
        video_files = list(candidates_path.glob("*.mp4")) + list(candidates_path.glob("*.mkv")) + list(candidates_path.glob("*.webm"))
        
        if not video_files:
            raise HTTPException(status_code=404, detail="No video files found in candidates folder")
        
        result = await fingerprint_client.generate_fpraw(
            source_path=str(candidates_path),
            output_path=str(fpraw_output),
            parallel=3
        )
        
        return {
            "success": True,
            "message": f"Generated fingerprints for {len(video_files)} videos",
            "output_path": str(fpraw_output),
            "video_count": len(video_files)
        }
        
    except Exception as e:
        logger.error(f"Fingerprint generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprint/project/create")
async def create_fingerprint_project(name: str):
    try:
        project_id = await fingerprint_client.create_project(name)
        
        return {
            "success": True,
            "project_id": project_id,
            "project_name": name
        }
        
    except Exception as e:
        logger.error(f"Project creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fingerprint/projects")
async def list_fingerprint_projects():
    try:
        result = await fingerprint_client.list_projects()
        return result
        
    except Exception as e:
        logger.error(f"Project list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprint/volume/create")
async def create_volume(volume_id: str, path: str, display_name: str):
    try:
        result = await fingerprint_client.create_volume(volume_id, path, display_name)
        return {"success": True, "volume_id": volume_id}
        
    except Exception as e:
        logger.error(f"Volume creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprint/archive/populate")
async def populate_archive(project_id: str):
    try:
        fpraw_path = Path("media_storage/fpraw")
        
        if not fpraw_path.exists():
            raise HTTPException(status_code=404, detail="FPRAW folder not found. Generate fingerprints first.")
        
        volume_id = "fpraw-media"
        await fingerprint_client.create_volume(
            volume_id=volume_id,
            path=str(fpraw_path.absolute()),
            display_name="FPRAW Media"
        )
        
        result = await fingerprint_client.add_media_to_project(
            project_id=project_id,
            volume_path=f"volume:fp-{volume_id}:/Archive"
        )
        
        return {
            "success": True,
            "message": "Archive populated successfully",
            "project_id": project_id
        }
        
    except Exception as e:
        logger.error(f"Archive population error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprint/search")
async def search_fingerprint(project_id: str, video_filename: str):
    try:
        fpraw_path = Path("media_storage/fpraw")
        query_file = fpraw_path / video_filename
        
        if not query_file.exists():
            query_file = query_file.with_suffix(".fpraw")
        
        if not query_file.exists():
            raise HTTPException(status_code=404, detail=f"FPRAW file not found: {video_filename}")
        
        volume_path = f"volume:fp-fpraw-media:/{query_file.name}"
        
        report_id = await fingerprint_client.generate_search_report(
            project_id=project_id,
            query_volume_path=volume_path,
            display_name=f"Search: {video_filename}"
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "project_id": project_id,
            "message": "Search report queued"
        }
        
    except Exception as e:
        logger.error(f"Fingerprint search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fingerprint/report/{project_id}/{report_id}")
async def get_fingerprint_report(project_id: str, report_id: str):
    try:
        status = await fingerprint_client.get_report_status(project_id, report_id)
        
        if status["status"] == "completed":
            report_data = await fingerprint_client.get_report_json(project_id, report_id)
            return {
                "success": True,
                "status": "completed",
                "report": report_data
            }
        else:
            return {
                "success": True,
                "status": status["status"],
                "progress": status.get("progress", "unknown")
            }
        
    except Exception as e:
        logger.error(f"Report retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fingerprint/candidates")
async def list_candidate_videos():
    try:
        candidates_path = Path("media_storage/candidates")
        
        if not candidates_path.exists():
            return {"videos": []}
        
        videos = []
        for video_file in candidates_path.glob("*"):
            if video_file.suffix in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
                videos.append({
                    "filename": video_file.name,
                    "size": video_file.stat().st_size,
                    "created": datetime.fromtimestamp(video_file.stat().st_ctime).isoformat()
                })
        
        return {"videos": videos}
        
    except Exception as e:
        logger.error(f"List candidates error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fingerprint/database-videos")
async def list_database_videos():
    try:
        database_path = Path("media_storage/database")
        
        if not database_path.exists():
            return {"videos": []}
        
        videos = []
        for video_file in database_path.glob("*"):
            if video_file.suffix in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
                videos.append({
                    "filename": video_file.name,
                    "size": video_file.stat().st_size,
                    "created": datetime.fromtimestamp(video_file.stat().st_ctime).isoformat()
                })
        
        return {"videos": videos, "count": len(videos)}
        
    except Exception as e:
        logger.error(f"List database videos error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprint/process-video")
async def process_video_simple(filename: str):
    try:
        logger.info(f"Processing YouTube video: {filename}")
        
        video_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        database_path = Path("media_storage/database")
        candidates_path = Path("media_storage/candidates")
        candidate_file = candidates_path / filename
        
        if not candidate_file.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
        
        if not database_path.exists() or not list(database_path.glob("*.mp4")):
            raise HTTPException(status_code=404, detail="No database videos found")
        
        logger.info("Checking REST server status...")
        if not fingerprint_client.check_rest_server():
            logger.info("REST server not running. Starting it now...")
            if not fingerprint_client.start_rest_server():
                raise HTTPException(status_code=500, detail="Failed to start REST server")
        else:
            logger.info("REST server already running")
        
        fpraw_root = Path(f"media_storage/fpraw/{video_name}")
        fpraw_database_folder = fpraw_root / "database"
        fpraw_candidate_folder = fpraw_root / "candidates"
        
        fpraw_database_folder.mkdir(parents=True, exist_ok=True)
        fpraw_candidate_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating FPRAW for ALL database videos...")
        await fingerprint_client.generate_fpraw(
            source_path=str(database_path.absolute()),
            output_path=str(fpraw_database_folder.absolute()),
            parallel=5
        )
        
        database_fpraws = list(fpraw_database_folder.glob("*.fpraw"))
        logger.info(f"Generated {len(database_fpraws)} database FPRAWs")
        
        if not database_fpraws:
            raise HTTPException(status_code=500, detail="Failed to generate database FPRAWs")
        
        logger.info(f"Generating FPRAW for candidate: {filename}")
        
        # Use a temp folder so folderwatch only processes this one video
        temp_source = Path("media_storage/temp_single_video")
        temp_source.mkdir(parents=True, exist_ok=True)
        
        import shutil
        temp_video = temp_source / filename
        shutil.copy2(candidate_file, temp_video)
        
        try:
            await fingerprint_client.generate_fpraw(
                source_path=str(temp_source.absolute()),
                output_path=str(fpraw_candidate_folder.absolute()),
                parallel=1
            )
        finally:
            shutil.rmtree(temp_source, ignore_errors=True)
        
        candidate_fpraw = fpraw_candidate_folder / f"{filename}.fpraw"
        if not candidate_fpraw.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate candidate FPRAW for {filename}"
            )
        
        logger.info(f"Generated candidate FPRAW: {candidate_fpraw.name}")
        
        # Volume ID must be lowercase: [a-z][a-z0-9-]{0,62}
        volume_id = f"{video_name.lower().replace('_', '-')}-trial"
        volume_name = f"fp-{volume_id}"
        
        logger.info(f"Creating volume: {volume_name}")
        
        try:
            await fingerprint_client.create_volume(
                volume_id=volume_id,
                path=str(fpraw_root.absolute()),
                display_name=f"FPRAW {video_name}"
            )
            logger.info(f"Created volume: {volume_name}")
        except Exception as e:
            logger.warning(f"Volume creation: {e} (may already exist)")
        
        logger.info(f"Creating Nablet project for {video_name}")
        project_id = await fingerprint_client.create_project(video_name)
        logger.info(f"Created project: {project_id}")
        
        logger.info(f"Adding ALL database FPRAWs to project...")
        database_volume_path = f"volume:{volume_name}:/database"
        await fingerprint_client.add_media_to_project(
            project_id=project_id,
            volume_path=database_volume_path
        )
        logger.info(f"Added database to project: {database_volume_path}")
        
        logger.info(f"Querying candidate against database...")
        candidate_volume_path = f"volume:{volume_name}:/candidates/{filename}.fpraw"
        
        report_id = await fingerprint_client.generate_search_report(
            project_id=project_id,
            query_volume_path=candidate_volume_path,
            display_name=f"Report_{video_name}"
        )
        logger.info(f"Report queued: {report_id}")
        
        logger.info(f"Waiting for report completion...")
        
        import asyncio
        status_info = None
        for i in range(60):
            await asyncio.sleep(1)
            status_info = await fingerprint_client.get_report_status(project_id, report_id)
            logger.info(f"Report status: {status_info.get('status')} ({i+1}s)")
            if status_info["status"] == "completed":
                break
        
        if status_info and status_info["status"] == "completed":
            logger.info(f"Retrieving report...")
            
            report_data = await fingerprint_client.get_report_json(project_id, report_id)
            
            result_folder = fpraw_root / "result"
            result_folder.mkdir(exist_ok=True)
            result_file = result_folder / f"report_{report_id}.json"
            
            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Report saved to: {result_file}")
            
            matches = []
            reference_videos = {}
            
            if report_data and isinstance(report_data, dict):
                if "reference" in report_data and isinstance(report_data["reference"], list):
                    for ref in report_data["reference"]:
                        video_id = ref.get("videoId")
                        
                        ref_files = ref.get("referenceFiles", [])
                        if ref_files and len(ref_files) > 0:
                            source_path = ref_files[0].get("sourcePath", "")
                            filename = Path(source_path).name if source_path else f"Video_{video_id}"
                        else:
                            filename = f"Video_{video_id}"
                        
                        reference_videos[video_id] = {
                            "filename": filename,
                            "hash": ref.get("referenceFileHash", "")
                        }
                    
                    logger.info(f"Reference videos: {reference_videos}")
                
                if "query" in report_data and len(report_data["query"]) > 0:
                    query_item = report_data["query"][0]
                    detected = query_item.get("detectedSegments", [])
                    
                    video_segment_count = {}
                    
                    for record in detected:
                        segments = record.get("segments", [])
                        for seg in segments:
                            vid_id = seg.get("referenceVideoId")
                            if vid_id not in video_segment_count:
                                video_segment_count[vid_id] = 0
                            video_segment_count[vid_id] += 1
                    
                    for vid_id, count in video_segment_count.items():
                        video_info = reference_videos.get(vid_id, {
                            "filename": f"Unknown_Video_{vid_id}",
                            "hash": ""
                        })
                        
                        matches.append({
                            "videoId": vid_id,
                            "filename": video_info["filename"],
                            "hash": video_info["hash"],
                            "matchCount": count
                        })
                    
                    matches.sort(key=lambda x: x["matchCount"], reverse=True)
                    
                    logger.info(f"Matched videos:")
                    for m in matches:
                        logger.info(f"  - {m['filename']}: {m['matchCount']} segments")
            
            total_match_count = sum(m['matchCount'] for m in matches)
            
            return {
                "success": True,
                "message": f"Found {total_match_count} matching segments" + (f" in {len(matches)} video(s)" if len(matches) > 0 else ""),
                "filename": filename,
                "video_name": video_name,
                "project_id": project_id,
                "report_id": report_id,
                "matchedVideos": matches,
                "report_saved": str(result_file)
            }
        else:
            return {
                "success": False,
                "message": "Search timeout or still processing",
                "status": status_info["status"] if status_info else "unknown",
                "project_id": project_id,
                "report_id": report_id
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process video error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprint/search-candidate")
async def search_candidate(filename: str, project_id: str):
    try:
        logger.info(f"Searching candidate: {filename} against project {project_id}")
        
        candidates_path = Path("media_storage/candidates")
        video_file = candidates_path / filename
        
        if not video_file.exists():
            raise HTTPException(status_code=404, detail=f"Candidate video not found: {filename}")
        
        video_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        fpraw_candidate_folder = Path("media_storage/fpraw/candidates") / video_name
        fpraw_candidate_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating FPRAW for candidate {filename}...")
        await fingerprint_client.generate_fpraw(
            source_path=str(candidates_path.absolute()),
            output_path=str(fpraw_candidate_folder.absolute()),
            parallel=1
        )
        
        logger.info("Starting REST server...")
        if not fingerprint_client.start_rest_server():
            raise HTTPException(status_code=500, detail="Failed to start REST server")
        
        try:
            query_path = f"volume:fp-media:/candidates/{video_name}"
            
            logger.info(f"Generating search report...")
            report_id = await fingerprint_client.generate_search_report(
                project_id=project_id,
                query_volume_path=query_path,
                display_name=f"Search_{video_name}"
            )
            
            import asyncio
            status_info = None
            for i in range(30):
                await asyncio.sleep(1)
                status_info = await fingerprint_client.get_report_status(project_id, report_id)
                logger.info(f"Report status: {status_info['status']}")
                if status_info["status"] == "completed":
                    break
            
            if status_info and status_info["status"] == "completed":
                logger.info("Report completed, fetching results...")
                report_data = await fingerprint_client.get_report_json(project_id, report_id)
                
                matches = []
                if report_data and isinstance(report_data, dict):
                    if "query" in report_data and len(report_data["query"]) > 0:
                        detected = report_data["query"][0].get("detectedSegments", [])
                        for item in detected:
                            record_nr = item.get("recordNr", 0)
                            ref_video_id = item.get("referenceVideoId", "unknown")
                            segments = item.get("segments", [])
                            if segments:
                                total_dist = sum(seg.get("dist0", 1.0) for seg in segments)
                                avg_similarity = int((1.0 - (total_dist / len(segments))) * 100)
                                
                                matches.append({
                                    "record_id": record_nr,
                                    "reference_video_id": ref_video_id,
                                    "similarity": avg_similarity,
                                    "segments": len(segments),
                                    "segment_details": segments[:5]
                                })
                
                return {
                    "success": True,
                    "candidate_video": filename,
                    "matches": matches,
                    "match_count": len(matches),
                    "message": f"Found {len(matches)} matches in database",
                    "raw_report": report_data
                }
            else:
                return {
                    "success": False,
                    "message": "Search timeout or still processing",
                    "status": status_info["status"] if status_info else "unknown"
                }
        
        finally:
            logger.info("Stopping REST server...")
            fingerprint_client.stop_rest_server()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Candidate search error: {e}")
        try:
            fingerprint_client.stop_rest_server()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


class KeywordRequest(BaseModel):
    video_url:  str  = Field(..., description="YouTube URL (or any yt-dlp supported URL) to download and analyze")
    run_asr:    bool = Field(True,  description="Run speech recognition")
    run_vlm:    bool = Field(True,  description="Run visual analysis (requires Ollama + qwen3-vl:4b)")
    language:   str  = Field("auto", description="ASR language hint: auto, en, de, fr, ...")


_keyword_jobs: dict[str, PipelineStatus] = {}


@app.post("/api/keywords/analyze")
async def start_keyword_analysis(request: KeywordRequest):
    raw_input = request.video_url.strip().strip('"').strip("'")
    if not raw_input:
        raise HTTPException(status_code=422, detail="video_url is required")

    job_id   = f"kw_{int(datetime.now().timestamp() * 1000)}"
    status   = PipelineStatus()
    proc_dir = Path(f"media_storage/keywords/{job_id}")

    _keyword_jobs[job_id] = status

    async def _run():
        try:
            if raw_input.startswith(("http://", "https://", "www.", "youtu")):
                status.update("download", f"Downloading video from URL...")
                try:
                    video_path = await video_downloader.download_candidate(
                        url=raw_input,
                        video_id=job_id,
                        title=job_id,
                    )
                    status.update("download", f"Downloaded: {video_path.name}")
                except Exception as exc:
                    status.fail(f"Download failed: {exc}")
                    return
            else:
                video_path = Path(raw_input)
                if not video_path.is_absolute():
                    for folder in ("media_storage/candidates", "media_storage/candidate"):
                        candidate = Path(folder) / raw_input
                        if candidate.exists():
                            video_path = candidate
                            break

                if not video_path.exists():
                    status.fail(f"File not found: {raw_input}")
                    return

            await run_keyword_pipeline(
                video_path=str(video_path),
                processing_dir=str(proc_dir),
                run_asr=request.run_asr,
                run_vlm=request.run_vlm,
                language=request.language,
                status=status,
            )

        except Exception as exc:
            logger.error(f"Keyword pipeline error [{job_id}]: {exc}\n{traceback.format_exc()}")
            if not status.done:
                status.fail(str(exc))

    asyncio.create_task(_run())
    logger.info(f"Keyword job {job_id} started for: {raw_input[:80]}")

    return {
        "success": True,
        "job_id":  job_id,
        "message": f"Analysis started",
        "input":   raw_input[:120],
    }


@app.get("/api/keywords/status/{job_id}")
async def get_keyword_status(job_id: str):
    status = _keyword_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return {
        "job_id":  job_id,
        "current": status.current,
        "steps":   status.steps,
        "done":    status.done,
        "error":   status.error,
        "result":  status.result,
    }


@app.get("/api/keywords/stream/{job_id}")
async def stream_keyword_progress(job_id: str):
    status = _keyword_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_generator():
        for step_entry in list(status.steps):
            yield f"data: {json.dumps(step_entry)}\n\n"

        if status.done:
            payload = {"step": "done", "result": status.result, "error": status.error}
            yield f"data: {json.dumps(payload)}\n\n"
            return

        queue: asyncio.Queue = asyncio.Queue()

        def on_update(entry: dict):
            queue.put_nowait(entry)

        status._callbacks.append(on_update)

        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(entry)}\n\n"
                    if entry.get("step") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    # Keep connection alive while pipeline runs
                    yield ": heartbeat\n\n"
        finally:
            status._callbacks.remove(on_update)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/keywords/result/{job_id}")
async def get_keyword_result(job_id: str):
    status = _keyword_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not status.done:
        raise HTTPException(status_code=202, detail="Job still running")
    if status.error:
        raise HTTPException(status_code=500, detail=status.error)
    return {"success": True, "job_id": job_id, **status.result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
