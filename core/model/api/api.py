from typing import Any, overload

from fastapi import WebSocket

from .base import At, Image, Record, Reply, Text, Video, MessageSegment
from .message import GroupMessageParams, GroupMessagePayload


class BotApi:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    @overload
    async def send_group_msg(
        self, group_id: str | int, *, message_segment: list[MessageSegment]
    ) -> None: ...

    @overload
    async def send_group_msg(
        self,
        group_id: str | int,
        *,
        text: str,
        at: int | str,
        image: str,
        reply: str | int,
        video: str,
        record: str,
    ) -> None: ...

    async def send_group_msg(
        self,
        group_id: str | int,
        *,
        message_segment: list[MessageSegment] | None = None,
        text: str | None = None,
        at: int | str | None = None,
        image: str | None = None,
        reply: str | int | None = None,
        video: str | None = None,
        record: str | None = None,
    ) -> None:
        mapping: list[tuple[Any, type[MessageSegment]]] = [
            (text, Text),
            (at, At),
            (image, Image),
            (reply, Reply),
            (video, Video),
            (record, Record),
        ]
        if message_segment is None:
            message_segment = []
            for value, cls in mapping:
                if value is not None:
                    message_segment.append(cls.new(value))
        payload = GroupMessagePayload(
            params=GroupMessageParams(group_id=group_id, message=message_segment)
        )
        await self.websocket.send_text(payload.model_dump_json())
