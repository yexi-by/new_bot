import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await websocket.accept()
    try:
        while True:
            data_dict = await websocket.receive_json()
            file_path = "debug.jsonl"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            message = data_dict.get("message", [])
            if not message:
                continue
            text = message[0]["data"]["text"]
            if text == "测试":
                payload = {
                    "action": "send_group_msg",
                    "params": {"group_id": 1049726313, "message": "大家好！"},
                }
                await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        print(f"用户 {client_id} 已断开连接")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
