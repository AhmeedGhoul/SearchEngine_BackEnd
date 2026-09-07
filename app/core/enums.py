from enum import Enum


class Platform(str, Enum):
    YOUTUBE = "youtube"


class CandidateStatus(str, Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    DOWNLOAD_FAILED = "download_failed"


class VideoDuration(str, Enum):
    ANY = "any"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SearchOrder(str, Enum):
    DATE = "date"
    RATING = "rating"
    RELEVANCE = "relevance"
    TITLE = "title"
    VIEW_COUNT = "viewCount"


class VideoType(str, Enum):
    ANY = "any"
    VIDEO = "video"
    CHANNEL = "channel"
    PLAYLIST = "playlist"


class SafeSearch(str, Enum):
    NONE = "none"
    MODERATE = "moderate"
    STRICT = "strict"


class VideoDefinition(str, Enum):
    ANY = "any"
    HIGH = "high"
    STANDARD = "standard"


class VideoDimension(str, Enum):
    ANY = "any"
    TWO_D = "2d"
    THREE_D = "3d"


class VideoCaption(str, Enum):
    ANY = "any"
    CLOSED_CAPTION = "closedCaption"
    NONE = "none"


class VideoLicense(str, Enum):
    ANY = "any"
    CREATIVE_COMMON = "creativeCommon"
    YOUTUBE = "youtube"


class EventType(str, Enum):
    ANY = "any"
    COMPLETED = "completed"
    LIVE = "live"
    UPCOMING = "upcoming"
