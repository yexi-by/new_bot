from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Literal
from _decorators import retry_policy
from config import LLMConfig


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    text: str | None = None
    image: bytes | None = None

    @model_validator(mode="after")
    def check_at_least_one(self):
        if self.text is None and self.image is None:
            raise ValueError("必须提供 text 或 image")
        return self


class LLMProvider(ABC):
    """
    所有 AI 服务商必须继承的基类
    """

    @abstractmethod
    async def get_ai_response(
        self,
        messages: list,
        model: str,
        **kwargs,
    ) -> str:
        pass


class LLMProviderWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str
    provider: LLMProvider


class ResilientLLMProvider(LLMProvider):
    def __init__(self, inner_provider: LLMProvider, llm_config: LLMConfig):
        self.inner_provider = inner_provider
        self.llm_config = llm_config

    async def get_ai_response(
        self, messages: list[ChatMessage], model: str, **kwargs
    ) -> str:
        @retry_policy(
            retry_count=self.llm_config.retry_count,
            retry_delay=self.llm_config.retry_delay,
        )
        async def _call_impl():
            return await self.inner_provider.get_ai_response(
                messages=messages, model=model, **kwargs
            )

        return await _call_impl()
