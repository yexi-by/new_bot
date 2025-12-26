from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from dishka.integrations.fastapi import FromDishka, setup_dishka, inject
from dishka import AsyncContainer
from typing import Callable
import secrets
from core.model.api import BotApi
from .event import get_event
import uvicorn
from log import get_logger

logger = get_logger(__name__)


class NapCat:
    def __init__(self, container: AsyncContainer) -> None:
        self.app = FastAPI()
        self.container = container
        setup_dishka(self.container, self.app)
        self._register_routes()

    async def _check_auth_token(
        self, websocket: WebSocket, token: str = "adm12345"
    ) -> None:
        headers = websocket.headers
        auth_header = headers.get("authorization", "")
        expected_header = "Bearer " + token
        if not secrets.compare_digest(auth_header, expected_header):  # 防时序攻击
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise ValueError

    def _register_routes(self):
        @self.app.websocket("/ws")
        @inject
        async def websocket_endpoint(
            websocket: WebSocket,
            bot_factory: FromDishka[Callable[[WebSocket], BotApi]],
        ) -> None:
            try:
                await self._check_auth_token(websocket=websocket)
                await websocket.accept()
            except ValueError:
                logger.error("token错误")
                return
            
            bot = bot_factory(websocket)
            try:
                while True:
                    data_dict = await websocket.receive_json()
                    event = get_event(msg_dict=data_dict)
            except WebSocketDisconnect:
                pass

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)
