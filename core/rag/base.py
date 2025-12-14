from abc import ABC, abstractmethod
from config import EmbeddingConfig
import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class EmbeddingProvider(ABC):
    """
    所有 Embedding 服务商必须继承的基类
    """

    @abstractmethod
    async def get_vector(
        self,
        model_name: str,
        input_text: str,
        **kwargs,
    ) -> str:
        pass


class ResilientLLMProvider(EmbeddingProvider):
    def __init__(
        self, inner_provider: EmbeddingProvider, embedding_config: EmbeddingConfig
    ):
        self.inner_provider = inner_provider
        self.embedding_config = embedding_config

    async def get_vector(self, model_name: str, input_text: str, **kwargs) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.embedding_config.retry_count),
            wait=wait_exponential(
                multiplier=1, min=self.embedding_config.retry_delay, max=10
            ),
            retry=retry_if_exception_type(
                (
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                )
            ),
        ):
            with attempt:
                response = await self.inner_provider.get_vector(
                    model_name=model_name, input_text=input_text, **kwargs
                )
                return response

        raise RuntimeError("所有重试后仍未能获取AI响应")
