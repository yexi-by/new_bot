from typing import Literal
from pydantic import BaseModel
from .base import MessageSegment


class PrivateMessageParams(BaseModel):
    user_id: str | int
    message: list[MessageSegment]


class PrivateMessagePayload(BaseModel):
    action: Literal["send_private_msg"] = "send_private_msg"
    params: PrivateMessageParams


class GroupMessageParams(BaseModel):
    group_id: str | int
    message: list[MessageSegment]


class GroupMessagePayload(BaseModel):
    action: Literal["send_group_msg"] = "send_group_msg"
    params: GroupMessageParams
