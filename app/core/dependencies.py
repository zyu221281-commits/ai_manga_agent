"""FastAPI dependency injection module.

Provides async DB sessions and shared singleton access.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_timeout=5,  # 取连接最多等 5 秒，避免池耗尽卡死
    connect_args={
        "connect_timeout": 3,  # psycopg3/libpq TCP 连接超时：3 秒不上就快速失败
    },
)

_async_session_factory = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an AsyncSession, closed after request."""
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncSession:
    """Convenience: returns a session directly (for scripts / background tasks)."""
    return _async_session_factory()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Returns the session factory for background task usage."""
    return _async_session_factory
