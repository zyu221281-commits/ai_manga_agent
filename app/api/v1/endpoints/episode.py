"""Episode management endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{episode_id}")
async def get_episode(episode_id: str):
    """Get episode status and details."""
    return {"episode_id": episode_id, "status": "pending"}


@router.post("/{episode_id}/retry")
async def retry_episode(episode_id: str):
    """Retry a failed episode."""
    return {"episode_id": episode_id, "status": "retrying"}


@router.get("/{episode_id}/assets")
async def list_episode_assets(episode_id: str):
    """List all assets for an episode."""
    return {"episode_id": episode_id, "assets": []}
