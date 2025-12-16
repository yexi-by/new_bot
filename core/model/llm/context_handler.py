from typing import Literal
from base import ChatMessage


class ContextStateMachine:
    """上下文管理"""

    def __init__(self, system_prompt: str, max_context_length: int) -> None:
        if max_context_length < 2:
            raise ValueError(f"最大上下文必须大于1,当前设置: {max_context_length}")
        self.system_prompt = ChatMessage(role="system", text=system_prompt)
        self.messages_lst = [self.system_prompt]
        self.max_context_length = max_context_length

    def add_msg(self, msg: ChatMessage) -> None:
        self.messages_lst.append(msg)
        if len(self.messages_lst) > self.max_context_length:
            del self.messages_lst[1:3]

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
