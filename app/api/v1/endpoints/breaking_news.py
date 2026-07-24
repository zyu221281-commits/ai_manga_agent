"""Breaking news / trending topic endpoints for content selection."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/hot")
async def get_hot_topics():
    """Get trending topics for content inspiration."""
    return {"topics": []}


@router.get("/trending")
async def get_trending_genres():
    """Get trending manga/comic genres."""
    return {"genres": []}
