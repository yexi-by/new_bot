from dishka import Provider, provide, Scope
from config import Settings, ModelParameterManager
from llm import LLMHandler


class MyProvider(Provider):
    @provide(scope=Scope.APP)
    def get_config(self) -> Settings:
        return Settings()

    @provide(scope=Scope.APP)
    def get_llm_setting(self, setting: Settings) -> ModelParameterManager:
        llm_settings = setting.llm_settings
        return ModelParameterManager(llm_settings=llm_settings)

    @provide(scope=Scope.APP)
    def get_llm_handler(
        self, settings: Settings, model_parameter_manager: ModelParameterManager
    ) -> LLMHandler:
        return LLMHandler(
            settings=settings, model_parameter_manager=model_parameter_manager
        )
