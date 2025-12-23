import tomllib
import httpx
from dishka import Provider, Scope, provide
from config import Settings
from .model.llm import LLMHandler
from .model.rag import SiliconFlowEmbedding, SearchVectors


class MyProvider(Provider):
    @provide(scope=Scope.APP)
    def get_config(self) -> Settings:
        with open("config/config.toml", "rb") as f:
            toml_data = tomllib.load(f)
        return Settings(**toml_data)

    @provide(scope=Scope.APP)
    def get_llm_handler(self, settings: Settings) -> LLMHandler:
        return LLMHandler(settings=settings)

    @provide(scope=Scope.APP)
    def get_siliconflow_embedding(self, settings: Settings) -> SiliconFlowEmbedding:
        client = httpx.AsyncClient()
        return SiliconFlowEmbedding(
            client=client, embedding_config=settings.embedding_settings
        )

    @provide(scope=Scope.APP)
    def get_search_vectors(self, settings: Settings) -> SearchVectors:
        return SearchVectors(directory=settings.faiss_file_location)
