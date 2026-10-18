from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag_dev_password"

    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"
    embedding_dimensions: int = 768

    chunk_size: int = 1000
    chunk_overlap: int = 150

    local_search_top_k: int = 5
    local_search_hops: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
