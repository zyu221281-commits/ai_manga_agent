"""Cost / budget dashboard endpoints."""

from fastapi import APIRouter, Depends

from app.core.dependencies import get_db_session
from app.services.cost_tracker import CostTracker

router = APIRouter()


@router.get("/dashboard")
async def cost_dashboard(session=Depends(get_db_session)):
    """Get budget dashboard with daily / monthly / per-episode metrics."""
    tracker = CostTracker(session)
    return await tracker.dashboard()


@router.get("/episode/{episode_id}")
async def episode_cost(episode_id: str, session=Depends(get_db_session)):
    """Get cost breakdown for a specific episode."""
    tracker = CostTracker(session)
    cost = await tracker.episode_cost(episode_id)
    return {"episode_id": episode_id, "cost_usd": cost}


@router.get("/series/{series_id}")
async def series_cost(series_id: str, session=Depends(get_db_session)):
    """Get total cost for a series."""
    tracker = CostTracker(session)
    cost = await tracker.series_cost(series_id)
    return {"series_id": series_id, "cost_usd": cost}
