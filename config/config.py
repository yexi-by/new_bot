from pydantic import BaseModel
from pydantic_settings import BaseSettings
import tomllib


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    llm_name: str
    model_name: str
    provider_type: str
    retry_count: int
    retry_delay: int


class EmbeddingConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    provider_type: str


class Settings(BaseSettings):
    llm_settings: list[LLMConfig]
    embedding_settings: list[EmbeddingConfig]

    def __init__(self):
        data = self.load_config()
        super().__init__(**data)

    def load_config(self):
        with open("config.toml", "rb") as f:
            data = tomllib.load(f)
        return data


class ModelParameterManager:
    def __init__(self, llm_settings: list[LLMConfig], llm_name: str = "gemini") -> None:
        self.llm_settings = llm_settings
        self.llm_name = llm_name

    def switch_model(self, llm_name: str) -> None:
        llm_names_lst = [setting.llm_name for setting in self.llm_settings]
        if llm_name not in llm_names_lst:
            raise ValueError(f"设置的模型不在配置中,当前模型列表为{llm_names_lst}")
        self.llm_name = llm_name
