from dishka import Provider, provide, Scope
from config import Settings
from .model.llm import LLMHandler


class MyProvider(Provider):
    @provide(scope=Scope.APP)
    def get_config(self) -> Settings:
        return Settings()

    @provide(scope=Scope.APP)
    def get_llm_handler(self, settings: Settings) -> LLMHandler:
        return LLMHandler(settings=settings)
