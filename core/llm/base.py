from abc import ABC, abstractmethod
from schemas import ChatMessage




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

