from abc import ABC, abstractmethod
from typing import Literal

from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict, model_validator
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    retry_if_result,
)

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
    model_vendors:str
    provider: LLMProvider


class ResilientLLMProvider(LLMProvider):
    def __init__(self, inner_provider: LLMProvider, llm_config: LLMConfig):
        self.inner_provider = inner_provider
        self.llm_config = llm_config

    async def get_ai_response(
        self, messages: list[ChatMessage], model: str, **kwargs
    ) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.llm_config.retry_count),
            wait=wait_exponential(
                multiplier=1, min=self.llm_config.retry_delay, max=10
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                )
            )
            | retry_if_result(lambda x: not x),
        ):
            with attempt:
                response = await self.inner_provider.get_ai_response(
                    messages=messages, model=model, **kwargs
                )
                return response

        raise RuntimeError("所有重试后仍未能获取AI响应")
