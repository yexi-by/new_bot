from pydantic import BaseModel
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model_vendors: str  # 模型厂商
    provider_type: str
    retry_count: int
    retry_delay: int


class EmbeddingConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    provider_type: str
    retry_count: int
    retry_delay: int


class Settings(BaseSettings):
    llm_settings: list[LLMConfig] = []
    embedding_settings: EmbeddingConfig
    faiss_file_location: str = ""
