from abc import ABC, abstractmethod
from pydantic import BaseModel,ConfigDict, model_validator
from typing import Literal


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

    @abstractmethod
    def _format_chat_messages(self, messages: list[ChatMessage]) -> list | tuple:
        pass


class LLMProviderWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str
    provider:LLMProvider
