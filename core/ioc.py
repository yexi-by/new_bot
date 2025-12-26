import tomllib
from typing import Callable

import httpx
from dishka import Provider, Scope, provide
from fastapi import WebSocket

from config import Settings

from .model.api import BotApi
from .model.llm import LLMHandler
from .model.rag import SearchVectors, SiliconFlowEmbedding


class MyProvider(Provider):
    @provide(scope=Scope.APP)
    def get_config(self) -> Settings:
        with open("config/config.toml", "rb") as f:
            toml_data = tomllib.load(f)
        return Settings(**toml_data)

    @provide(scope=Scope.APP)
    def get_llm_handler(self, settings: Settings) -> LLMHandler:
        return LLMHandler.register_instance(settings=settings.llm_settings)

    @provide(scope=Scope.APP)
    def get_siliconflow_embedding(self, settings: Settings) -> SiliconFlowEmbedding:
        client = httpx.AsyncClient()
        return SiliconFlowEmbedding(
            client=client, embedding_config=settings.embedding_settings
        )

    @provide(scope=Scope.APP)
    async def get_search_vectors(self, settings: Settings) -> SearchVectors:
        return await SearchVectors.create_from_directory(
            directory=settings.faiss_file_location
        )

    @provide(scope=Scope.SESSION)
    def get_bot_api(self) -> Callable[[WebSocket], BotApi]:
        def factory(websocket: WebSocket):
            return BotApi(websocket=websocket)
        return factory
