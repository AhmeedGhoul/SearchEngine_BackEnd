from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NABLET"
    debug: bool = True
    
    # YouTube API
    youtube_api_key: str | None = None
    
    # Storage
    media_storage_path: str = "media_storage"
    candidates_folder: str = "candidates"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


settings = Settings()
