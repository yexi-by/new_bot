from pydantic import BaseModel,model_validator
from pydantic_settings import BaseSettings
import tomllib
from typing import Any


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model_vendors: str #模型厂商
    model_name: str
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
    llm_settings: list[LLMConfig]=[]
    embedding_settings: EmbeddingConfig|None=None
    
    @model_validator(mode='before')
    @classmethod
    def load_config_from_toml(cls, data: Any) -> Any:
        if isinstance(data, dict) and data:
            return data
        with open("config.toml", "rb") as f:
            toml_data = tomllib.load(f)
            return toml_data
        return data
        
