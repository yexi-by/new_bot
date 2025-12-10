from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal


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

class ChatMessage(BaseModel):
    role: str
    text: str
    image: bytes



