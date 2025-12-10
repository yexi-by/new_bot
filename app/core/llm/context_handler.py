from base import 
from gemini_llm import GeMiniAIService
from openai_llm import OpenAIService
import asyncio


class ContextStateMachine:
    def __init__(
        self, system_prompt: str, current_container
    ) -> None:  # current_container理论上是个容器,后面可以根据需要变成一个函数(待实现)
        self.system_prompt = system_prompt
        self.current_container = current_container
        self.messages: list[OpenAIMessage | GeMiniMessage] = []
        self.raw_messages: list[dict] = []
        self._monitor_task = asyncio.create_task(self.contextMonitor())

    def __del__(self) -> None:
        try:
            self._monitor_task.cancel()
        except RuntimeError:
            pass

    def _convert_context_format(
        self, message_obj: type[GeMiniMessage] | type[OpenAIMessage]
    ) -> list[OpenAIMessage | GeMiniMessage]:
        """转化为上下文格式"""
        if isinstance(self.messages[0], message_obj):
            return self.messages
        strategies = {
            OpenAIMessage: lambda m: OpenAIMessage(
                role=m["role"], content=m["content"]
            ),
            GeMiniMessage: lambda m: GeMiniMessage(
                role=m["role"], parts=[{"text": m["content"]}]
            ),
        }
        converter = strategies.get(message_obj)
        if not converter:
            raise ValueError(f"不支持的消息类型: {message_obj}")
        new_messages = [converter(msg) for msg in self.raw_messages]
        return new_messages

    async def contextMonitor(self) -> None:
        """监控实例状态"""
        context_map = {OpenAIService: OpenAIMessage, GeMiniAIService: GeMiniMessage}
        while True:
            await asyncio.sleep(0.1)
            if not self.raw_messages or not self.messages:
                continue
            current_obj = self.current_container.llm_obj
            current_type = type(current_obj)
            self.messages = self._convert_context_format(context_map[current_type])
    
    async def 
