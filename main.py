import json
import secrets
from typing import Callable
import uvicorn
from dishka import make_async_container
from dishka.integrations.fastapi import FromDishka, setup_dishka, inject
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from core.ioc import MyProvider
from core.model.api import BotApi
from log import setup_exception_handler, logger

setup_exception_handler()

app = FastAPI()
container = make_async_container(MyProvider())
setup_dishka(container, app)


@app.websocket("/ws/{client_id}")
@inject
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: int,
    bot_factory: FromDishka[Callable[[WebSocket], BotApi]],
) -> None:
    headers = websocket.headers
    auth_header = headers.get("authorization", "")
    my_token = "adm123456"
    expected_header = "Bearer " + my_token
    if not secrets.compare_digest(auth_header, expected_header):  # 防时序攻击
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    logger.info(f"用户 {client_id} 已建立连接")
    bot = bot_factory(websocket)
    try:
        while True:
            data_dict = await websocket.receive_json()
            logger.info(data_dict)
            file_path = "debug.jsonl"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
    except WebSocketDisconnect:
        logger.info(f"用户 {client_id} 已断开连接")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
