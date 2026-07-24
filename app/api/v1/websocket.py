"""WebSocket endpoint for real-time pipeline progress."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/{episode_id}")
async def episode_progress(websocket: WebSocket, episode_id: str):
    """Stream episode generation progress in real-time."""
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({
                "episode_id": episode_id,
                "status": "running",
                "progress": 0.5,
                "message": "Processing...",
            })
    except WebSocketDisconnect:
        pass
