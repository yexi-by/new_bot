from google import genai
from google.genai import types
from openai import AsyncOpenAI
from gemini_llm import GeminiAIService
from openai_llm import OpenAIService
from base import ChatMessage,LLMProviderWrapper
from config import Settings, ModelParameterManager


class LLMHandler:
    def __init__(
        self, settings: Settings, model_parameter_manager: ModelParameterManager
    ) -> None:
        self.llm_settings = settings.llm_settings
        self.model_params = model_parameter_manager
        self.service_map: dict[str, LLMProviderWrapper] = {}
        self._register_instance()

    def _register_instance(self) -> None:
        """注册实例"""
        model_map = {
            "openai": lambda api_key, base_url: OpenAIService(
                client=AsyncOpenAI(api_key=api_key, base_url=base_url)
            ),
            "gemini": lambda api_key, base_url: GeminiAIService(
                client=genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(base_url=base_url)
                    if base_url
                    else None,
                )
            ),
        }
        for model_config in self.llm_settings:
            service_factory = model_map.get(model_config.provider_type)
            if service_factory is None:
                raise ValueError(f"未知的模型服务类型: {model_config.provider_type}")
            servicel = service_factory(model_config.api_key, model_config.base_url)
            self.service_map[model_config.llm_name] = LLMProviderWrapper(
                model_name=model_config.model_name, provider=servicel
            )

    async def get_ai_text_response(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> str:
        current_instance = self.service_map[self.model_params.llm_name]
        servicel = current_instance.provider
        model_name = current_instance.model_name
        ai_response = await servicel.get_ai_response(
            messages=messages, model=model_name
        )
        return ai_response
