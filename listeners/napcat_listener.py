from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
app = FastAPI()
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    # 1. 接受连接
    await websocket.accept()
    try:
        while True:
            # 2. 接收客户端发送的文本
            data = await websocket.receive_text()
            # 3. 发送消息回客户端
            await websocket.send_text(f"用户ID: {client_id} 说: {data}")
    except WebSocketDisconnect:
        print(f"用户 {client_id} 已断开连接")
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)