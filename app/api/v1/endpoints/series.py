"""Series CRUD endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_series():
    """List all series."""
    return {"series": [], "total": 0}


@router.get("/{series_id}")
async def get_series(series_id: str):
    """Get series details."""
    return {"series_id": series_id, "status": "draft"}


@router.post("/")
async def create_series():
    """Create a new series from creative brief."""
    return {"series_id": "new_id", "status": "created"}
