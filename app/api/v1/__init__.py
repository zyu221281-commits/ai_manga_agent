"""API v1 router - aggregates all endpoint routers."""

from fastapi import APIRouter

from app.api.v1.endpoints.admin import router as admin_router
from app.api.v1.endpoints.breaking_news import router as breaking_news_router
from app.api.v1.endpoints.cost import router as cost_router
from app.api.v1.endpoints.episode import router as episode_router
from app.api.v1.endpoints.series import router as series_router

router = APIRouter()
router.include_router(series_router, prefix="/series", tags=["Series"])
router.include_router(episode_router, prefix="/episode", tags=["Episode"])
router.include_router(cost_router, prefix="/cost", tags=["Cost"])
router.include_router(admin_router, prefix="/admin", tags=["Admin"])
router.include_router(breaking_news_router, prefix="/breaking", tags=["Breaking News"])
