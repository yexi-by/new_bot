from base import ChatMessage
from _decorators import sliding_context_window
from typing import Literal


class ContextStateMachine:
    """上下文管理"""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = ChatMessage(role="system", text=system_prompt)
        self.messages_lst = [self.system_prompt]

    @sliding_context_window(20)
    def add_msg(self, msg: ChatMessage) -> None:
        self.messages_lst.append(msg)

    def build_chatmessage(
        self,
        role: Literal["system", "user", "assistant"],
        text: str | None = None,
        image: bytes | None = None,
    ) -> None:
        if role == "system":  # 系统提示词理论上应该只有一份
            if not text or image:
                raise ValueError("系统提示词应该并且必须是字符串")
            self.messages_lst[0] = ChatMessage(
                role="system",
                text=text,
            )
        else:
            chatmessage = ChatMessage(role=role, text=text, image=image)
            self.add_msg(msg=chatmessage)
