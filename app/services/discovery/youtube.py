import re
import httpx
from datetime import datetime, timezone
from typing import Any

from app.services.discovery.base import DiscoveryProvider
from app.core.config import settings
from app.core.logging import get_logger
from app.core.enums import (
    Platform, VideoDuration, SearchOrder, VideoType, SafeSearch,
    VideoDefinition, VideoDimension, VideoCaption, VideoLicense, EventType
)

logger = get_logger(__name__)


class YouTubeAPIError(Exception):
    """Base YouTube API error"""
    pass


class YouTubeAPIAuthError(YouTubeAPIError):
    """Authentication/API key error"""
    pass


class YouTubeAPIRateLimitError(YouTubeAPIError):
    """Rate limit exceeded"""
    pass


class YouTubeAPIRequestError(YouTubeAPIError):
    """Invalid request parameters"""
    pass


class YouTubeAPITimeoutError(YouTubeAPIError):
    """Request timeout"""
    pass


class YouTubeDiscoveryProvider(DiscoveryProvider):
    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.youtube_api_key
        if not self.api_key:
            raise ValueError("YouTube API key is required")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, keywords: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        video_type = kwargs.get("video_type", VideoType.VIDEO)
        max_results = kwargs.get("max_results", 50)
        channel_ids = kwargs.get("channel_ids") or []

        if video_type == VideoType.CHANNEL:
            query = self._build_search_query(keywords, **kwargs)
            return await self._search_channels(query, **kwargs)

        query = self._build_search_query(keywords, **kwargs)

        logger.info(
            "YouTube search: query=%r channel_ids=%s max_results=%s",
            query,
            channel_ids,
            max_results,
        )

        has_video_filters = self._has_active_filters(**kwargs)

        if len(channel_ids) > 1:
            kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['max_results', 'channel_ids']}
            return await self._search_multiple_channels(
                query=query,
                channel_ids=channel_ids,
                max_results=max_results,
                has_filters=has_video_filters or bool(query),
                **kwargs_filtered
            )
        elif len(channel_ids) == 1:
            if not query and not has_video_filters:
                return await self._browse_channel_uploads(channel_ids[0], max_results)
            else:
                kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'max_results'}
                return await self._search_single_scope(
                    query=query,
                    channel_id=channel_ids[0],
                    max_results=max_results,
                    **kwargs_filtered
                )
        else:
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'max_results'}
            return await self._search_single_scope(
                query=query,
                channel_id=None,
                max_results=max_results,
                **kwargs_filtered
            )

    def _has_active_filters(self, **kwargs) -> bool:
        return any([
            kwargs.get("duration") and kwargs.get("duration") != VideoDuration.ANY,
            kwargs.get("video_definition") and kwargs.get("video_definition") != VideoDefinition.ANY,
            kwargs.get("video_dimension") and kwargs.get("video_dimension") != VideoDimension.ANY,
            kwargs.get("video_caption") and kwargs.get("video_caption") != VideoCaption.ANY,
            kwargs.get("video_license") and kwargs.get("video_license") != VideoLicense.ANY,
            kwargs.get("event_type") and kwargs.get("event_type") != EventType.ANY,
            kwargs.get("video_category_id"),
            kwargs.get("published_after"),
            kwargs.get("published_before"),
            kwargs.get("language"),
            kwargs.get("region"),
        ])

    async def _search_single_scope(
        self,
        query: str,
        channel_id: str | None,
        max_results: int,
        **kwargs
    ) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while len(all_results) < max_results:
            remaining = max_results - len(all_results)
            page_size = min(50, remaining)

            try:
                results = await self._search_page(
                    query=query,
                    max_results=page_size,
                    page_token=next_page_token,
                    channel_id=channel_id,
                    published_after=kwargs.get("published_after"),
                    published_before=kwargs.get("published_before"),
                    language=kwargs.get("language"),
                    region=kwargs.get("region"),
                    duration=kwargs.get("duration", VideoDuration.ANY),
                    order=kwargs.get("order", SearchOrder.DATE),
                    video_type=kwargs.get("video_type", VideoType.VIDEO),
                    safe_search=kwargs.get("safe_search", SafeSearch.MODERATE),
                    video_definition=kwargs.get("video_definition"),
                    video_dimension=kwargs.get("video_dimension"),
                    video_caption=kwargs.get("video_caption"),
                    video_license=kwargs.get("video_license"),
                    event_type=kwargs.get("event_type"),
                    video_category_id=kwargs.get("video_category_id"),
                )

                items = results.get("items", [])
                if not items:
                    break

                search_video_ids = [
                    item["id"]["videoId"]
                    for item in items
                    if item.get("id", {}).get("kind") == "youtube#video"
                    and item.get("id", {}).get("videoId")
                ]

                if not search_video_ids:
                    next_page_token = results.get("nextPageToken")
                    if not next_page_token:
                        break
                    continue

                metadata = await self._get_videos_metadata(search_video_ids)
                metadata_by_id = {video["id"]: video for video in metadata}

                for video_id in search_video_ids:
                    video = metadata_by_id.get(video_id)
                    if not video:
                        continue

                    all_results.append(self._normalize_video(video))
                    if len(all_results) >= max_results:
                        break

                next_page_token = results.get("nextPageToken")
                if not next_page_token:
                    break

            except httpx.TimeoutException as e:
                logger.error("YouTube API timeout: %s", e)
                raise YouTubeAPITimeoutError(f"Request timeout: {e}") from e

            except httpx.HTTPStatusError as e:
                self._handle_http_error(e)

            except httpx.HTTPError as e:
                logger.error("YouTube HTTP error: %s", e)
                raise YouTubeAPIError(f"Network error: {e}") from e

            except Exception as e:
                logger.exception("Unexpected YouTube search error")
                raise YouTubeAPIError(f"Search failed: {e}") from e

        return all_results[:max_results]

    async def _search_multiple_channels(
        self,
        query: str,
        channel_ids: list[str],
        max_results: int,
        has_filters: bool,
        **kwargs
    ) -> list[dict[str, Any]]:
        logger.info(f"Multi-channel search: {len(channel_ids)} channels, has_filters={has_filters}")

        per_channel_fetch = max(10, max_results * 2 // len(channel_ids))

        all_results = []
        seen_video_ids = set()

        for channel_id in channel_ids:
            try:
                kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'max_results'}
                
                if has_filters:
                    channel_results = await self._search_single_scope(
                        query=query,
                        channel_id=channel_id,
                        max_results=per_channel_fetch,
                        **kwargs_filtered
                    )
                else:
                    channel_results = await self._browse_channel_uploads(
                        channel_id,
                        per_channel_fetch
                    )

                for video in channel_results:
                    video_id = video.get("platform_video_id")
                    if video_id and video_id not in seen_video_ids:
                        seen_video_ids.add(video_id)
                        all_results.append(video)

            except Exception as e:
                logger.error(f"Error searching channel {channel_id}: {e}")
                continue

        order = kwargs.get("order", SearchOrder.DATE)
        all_results = self._sort_results(all_results, order)

        return all_results[:max_results]

    def _sort_results(self, results: list[dict[str, Any]], order: SearchOrder) -> list[dict[str, Any]]:
        if order == SearchOrder.DATE:
            return sorted(
                results,
                key=lambda x: x.get("published_at") or "",
                reverse=True
            )
        elif order == SearchOrder.VIEW_COUNT:
            return sorted(
                results,
                key=lambda x: x.get("view_count") or 0,
                reverse=True
            )
        elif order == SearchOrder.RATING:
            return sorted(
                results,
                key=lambda x: x.get("like_count") or 0,
                reverse=True
            )
        elif order == SearchOrder.TITLE:
            return sorted(
                results,
                key=lambda x: (x.get("title") or "").lower()
            )
        else:
            return results

    def _handle_http_error(self, error: httpx.HTTPStatusError):
        status = error.response.status_code

        if status == 429:
            raise YouTubeAPIRateLimitError("YouTube API rate limit exceeded.")

        if status == 403:
            try:
                error_json = error.response.json()
            except Exception:
                error_json = {}

            reason = (
                error_json.get("error", {})
                .get("errors", [{}])[0]
                .get("reason", "")
            )

            if reason == "quotaExceeded":
                raise YouTubeAPIRateLimitError("YouTube API quota exceeded.")

            raise YouTubeAPIAuthError("YouTube API key is invalid or forbidden.")

        if status == 400:
            raise YouTubeAPIRequestError("Invalid YouTube search parameters.")

        raise YouTubeAPIError(f"YouTube API HTTP {status}: {error}") from error

    async def _browse_channel_uploads(
        self,
        channel_id: str,
        max_results: int
    ) -> list[dict[str, Any]]:
        logger.info(f"Browsing uploads for channel: {channel_id}")

        try:
            params = {
                "part": "contentDetails",
                "id": channel_id,
                "key": self.api_key,
            }

            response = await self.client.get(f"{self.BASE_URL}/channels", params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get("items"):
                logger.warning(f"Channel not found: {channel_id}")
                return []

            uploads_playlist_id = (
                data["items"][0]
                .get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )

            if not uploads_playlist_id:
                logger.warning(f"No uploads playlist for channel: {channel_id}")
                return []

            all_video_ids = []
            next_page_token = None

            while len(all_video_ids) < max_results:
                remaining = max_results - len(all_video_ids)
                page_size = min(50, remaining)

                playlist_params = {
                    "part": "contentDetails",
                    "playlistId": uploads_playlist_id,
                    "maxResults": page_size,
                    "key": self.api_key,
                }

                if next_page_token:
                    playlist_params["pageToken"] = next_page_token

                response = await self.client.get(
                    f"{self.BASE_URL}/playlistItems",
                    params=playlist_params
                )
                response.raise_for_status()
                playlist_data = response.json()

                items = playlist_data.get("items", [])
                if not items:
                    break

                for item in items:
                    video_id = item.get("contentDetails", {}).get("videoId")
                    if video_id:
                        all_video_ids.append(video_id)
                        if len(all_video_ids) >= max_results:
                            break

                next_page_token = playlist_data.get("nextPageToken")
                if not next_page_token:
                    break

            if not all_video_ids:
                return []

            metadata = await self._get_videos_metadata(all_video_ids)
            metadata_by_id = {video["id"]: video for video in metadata}

            results = []
            for video_id in all_video_ids:
                video = metadata_by_id.get(video_id)
                if video:
                    results.append(self._normalize_video(video))

            return results

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            logger.exception(f"Unexpected error browsing channel: {e}")
            raise YouTubeAPIError(f"Failed to browse channel: {e}") from e

    async def _search_channels(self, query: str, **kwargs) -> list[dict[str, Any]]:
        max_results = kwargs.get('max_results', 50)
        all_results = []
        next_page_token = None

        while len(all_results) < max_results:
            try:
                remaining = max_results - len(all_results)
                page_size = min(50, remaining)

                results = await self._search_page(
                    query=query,
                    max_results=page_size,
                    page_token=next_page_token,
                    order=kwargs.get('order', SearchOrder.RELEVANCE),
                    video_type=VideoType.CHANNEL,
                    safe_search=kwargs.get('safe_search', SafeSearch.MODERATE),
                )

                if not results.get('items'):
                    break

                for item in results['items']:
                    all_results.append(self._normalize_channel(item))
                    if len(all_results) >= max_results:
                        break

                next_page_token = results.get('nextPageToken')
                if not next_page_token:
                    break

            except httpx.HTTPStatusError as e:
                self._handle_http_error(e)

        return all_results[:max_results]

    def _build_search_query(
        self,
        keywords: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        parts: list[str] = []

        if keywords:
            parts.extend(
                keyword.strip()
                for keyword in keywords
                if keyword and keyword.strip()
            )

        for phrase in kwargs.get("phrases") or []:
            phrase = phrase.strip()
            if phrase:
                parts.append(f'"{phrase}"')

        for hashtag in kwargs.get("hashtags") or []:
            hashtag = hashtag.strip()
            if not hashtag:
                continue
            if not hashtag.startswith("#"):
                hashtag = f"#{hashtag}"
            parts.append(hashtag)

        include_terms = [
            term.strip()
            for term in (kwargs.get("include_terms") or [])
            if term and term.strip()
        ]
        if include_terms:
            parts.append("(" + "|".join(include_terms) + ")")

        for term in kwargs.get("exclude_terms") or []:
            term = term.strip()
            if term:
                parts.append(f"-{term}")

        return " ".join(parts).strip()

    async def _search_page(
        self,
        query: str,
        max_results: int = 50,
        page_token: str | None = None,
        published_after: datetime | None = None,
        published_before: datetime | None = None,
        language: str | None = None,
        region: str | None = None,
        duration: VideoDuration = VideoDuration.ANY,
        order: SearchOrder = SearchOrder.RELEVANCE,
        channel_id: str | None = None,
        video_type: VideoType = VideoType.VIDEO,
        safe_search: SafeSearch = SafeSearch.MODERATE,
        video_definition: VideoDefinition | None = None,
        video_dimension: VideoDimension | None = None,
        video_caption: VideoCaption | None = None,
        video_license: VideoLicense | None = None,
        event_type: EventType | None = None,
        video_category_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "part": "snippet",
            "maxResults": max_results,
            "key": self.api_key,
        }

        if video_type != VideoType.ANY:
            params["type"] = video_type.value

        if query:
            params["q"] = query

        if page_token:
            params["pageToken"] = page_token

        if published_after:
            params["publishedAfter"] = (
                published_after.astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )

        if published_before:
            params["publishedBefore"] = (
                published_before.astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )

        if language:
            params["relevanceLanguage"] = language

        if region:
            params["regionCode"] = region.upper()

        if channel_id:
            params["channelId"] = channel_id

        if video_type == VideoType.VIDEO:
            if duration and duration != VideoDuration.ANY:
                params["videoDuration"] = duration.value

            if video_definition and video_definition != VideoDefinition.ANY:
                params["videoDefinition"] = video_definition.value

            if video_dimension and video_dimension != VideoDimension.ANY:
                params["videoDimension"] = video_dimension.value

            if video_caption and video_caption != VideoCaption.ANY:
                params["videoCaption"] = video_caption.value

            if video_license and video_license != VideoLicense.ANY:
                params["videoLicense"] = video_license.value

            if event_type and event_type != EventType.ANY:
                params["eventType"] = event_type.value

            if video_category_id:
                params["videoCategoryId"] = video_category_id

        if safe_search != SafeSearch.NONE:
            params["safeSearch"] = safe_search.value

        params["order"] = order.value

        logger.debug("YouTube API params: %s", {**params, "key": "***"})

        response = await self.client.get(f"{self.BASE_URL}/search", params=params)
        response.raise_for_status()
        return response.json()

    async def _get_videos_metadata(self, video_ids: list[str]) -> list[dict[str, Any]]:
        if not video_ids:
            return []

        params = {
            'part': 'snippet,contentDetails,statistics',
            'id': ','.join(video_ids),
            'key': self.api_key,
        }

        try:
            response = await self.client.get(f"{self.BASE_URL}/videos", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except httpx.TimeoutException as e:
            raise YouTubeAPITimeoutError(f"Metadata request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except httpx.HTTPError as e:
            raise YouTubeAPIError(f"Failed to fetch video metadata: {e}") from e

    def _normalize_channel(self, item: dict[str, Any]) -> dict[str, Any]:
        snippet = item.get('snippet', {})
        channel_id = item.get('id', {}).get('channelId')

        published_at = None
        if snippet.get('publishedAt'):
            try:
                published_at = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
            except Exception:
                pass

        return {
            'platform': Platform.YOUTUBE.value,
            'platform_video_id': channel_id,
            'channel_id': channel_id,
            'url': f'https://www.youtube.com/channel/{channel_id}',
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'channel_name': snippet.get('title', ''),
            'channel_url': f'https://www.youtube.com/channel/{channel_id}',
            'published_at': published_at.isoformat() if published_at else None,
            'thumbnail_url': self._get_best_thumbnail(snippet.get('thumbnails', {})),
            'duration_seconds': None,
            'view_count': None,
            'like_count': None,
            'comment_count': None,
            'definition': None,
            'dimension': None,
            'caption_available': False,
            'license': None,
        }

    def _normalize_video(self, item: dict[str, Any]) -> dict[str, Any]:
        snippet = item.get('snippet', {})
        content_details = item.get('contentDetails', {})
        statistics = item.get('statistics', {})
        video_id = item.get('id')

        published_at = None
        if snippet.get('publishedAt'):
            try:
                published_at = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
            except Exception:
                pass

        duration_seconds = None
        if content_details.get('duration'):
            duration_seconds = self._parse_duration(content_details['duration'])

        return {
            'platform': Platform.YOUTUBE.value,
            'platform_video_id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'channel_id': snippet.get('channelId'),
            'channel_name': snippet.get('channelTitle'),
            'channel_url': f"https://www.youtube.com/channel/{snippet.get('channelId')}" if snippet.get('channelId') else None,
            'published_at': published_at.isoformat() if published_at else None,
            'thumbnail_url': self._get_best_thumbnail(snippet.get('thumbnails', {})),
            'duration_seconds': duration_seconds,
            'view_count': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else None,
            'like_count': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else None,
            'comment_count': int(statistics.get('commentCount', 0)) if statistics.get('commentCount') else None,
            'definition': content_details.get('definition'),
            'dimension': content_details.get('dimension'),
            'caption_available': content_details.get('caption') == 'true',
            'license': content_details.get('licensedContent'),
        }

    def _get_best_thumbnail(self, thumbnails: dict[str, Any]) -> str | None:
        for quality in ['maxres', 'high', 'medium', 'default']:
            if quality in thumbnails:
                return thumbnails[quality].get('url')
        return None

    def _parse_duration(self, duration_str: str) -> int | None:
        try:
            s = duration_str.replace('PT', '')
            hours = int(h.group(1)) if (h := re.search(r'(\d+)H', s)) else 0
            minutes = int(m.group(1)) if (m := re.search(r'(\d+)M', s)) else 0
            seconds = int(sec.group(1)) if (sec := re.search(r'(\d+)S', s)) else 0
            return hours * 3600 + minutes * 60 + seconds
        except Exception:
            return None

    async def close(self):
        await self.client.aclose()
