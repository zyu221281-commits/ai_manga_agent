"""FastAPI application entry point for AI Manga Agent.""".

Exposes:
- /health            Health check
- /api/v1/series/*   Series CRUD
- /api/v1/episode/*  Episode management
- /api/v1/task/*     Task / pipeline triggers
- /api/v1/cost/*     Budget dashboard
- /api/v1/review/*   Human-in-the-loop review dashboard
- /api/v1/admin/*    Admin / system status
- /api/v1/audit/*    Audit log query
- /ws                WebSocket for real-time progress
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router as v1_router
from app.api.v1.websocket import router as ws_router
from app.core.config import settings
from app.observability import logger, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.configure_logging(settings.LOG_LEVEL)
    metrics.init_metrics()
    import logging as std_logging
    std_logging.getLogger("app").info("AI Manga Agent starting (env=%s)", settings.APP_ENV)

    # 生产环境安全检查
    security_warnings = settings.validate_production_security()
    for w in security_warnings:
        std_logging.getLogger("app").warning(w)

    # 从 Redis 加载历史记忆到内存 L1 缓存（fire-and-forget，失败不阻塞启动）
    try:
        from app.services.long_term_memory import long_term_memory
        await long_term_memory.load_from_redis()
    except Exception as e:
        std_logging.getLogger("app").warning("Failed to load long_term_memory from Redis: %s", e)

    yield
    std_logging.getLogger("app").info("AI Manga Agent shutting down")


app = FastAPI(
    title="AI Manga Agent",
    description="Vertical manga-drama factory with 6-agent pipeline + RPA automation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置：修复 allow_origins=["*"] + allow_credentials=True 冲突
# - 配置了 CORS_ORIGINS: 使用指定白名单 + 允许凭证
# - 未配置 CORS_ORIGINS 且为开发环境: 允许所有源但不携带凭证
# - 未配置 CORS_ORIGINS 且为生产环境: 不添加 CORS 中间件（拒绝跨域）
_cors_origins = settings.cors_origins_list
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
elif settings.APP_ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # 与 allow_origins=["*"] 配合时不携带凭证
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(v1_router, prefix="/api/v1")
app.include_router(ws_router)


@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "ok", "version": "1.0.0", "env": settings.APP_ENV}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
