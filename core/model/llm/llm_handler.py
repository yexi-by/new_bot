from google import genai
from google.genai import types
from openai import AsyncOpenAI
from gemini_llm import GeminiAIService
from openai_llm import OpenAIService
from base import ChatMessage, LLMProviderWrapper, ResilientLLMProvider
from config import Settings


class LLMHandler:
    def __init__(self, settings: Settings) -> None:
        self.llm_settings = settings.llm_settings
        self.services: list[LLMProviderWrapper] = []
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
            factory = model_map.get(model_config.provider_type)
            if factory is None:
                raise ValueError(f"未知的模型服务类型: {model_config.provider_type}")
            raw_service = factory(model_config.api_key, model_config.base_url)
            safe_service = ResilientLLMProvider(
                inner_provider=raw_service, llm_config=model_config
            )
            wrapper = LLMProviderWrapper(
                model_vendors=model_config.model_vendors,
                provider=safe_service,
            )
            self.services.append(wrapper)

    async def get_ai_text_response(
        self,
        messages: list[ChatMessage],
        model_vendors: str,
        model_name: str,
        **kwargs,
    ) -> str:
        for llm in self.services:
            if llm.model_vendors != model_vendors:
                continue
            return await llm.provider.get_ai_response(
                messages=messages, model=model_name
            )
        raise ValueError(f"未定义的服务商名:{model_vendors}")
