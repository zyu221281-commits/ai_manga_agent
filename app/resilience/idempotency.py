"""幂等锁（Redis SET NX EX 分布式锁）

 韧性加固：episode_id 级别防重投。
同一 episode_id 在锁有效期内不会被重复执行。
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from app.services.cache import cache

logger = logging.getLogger(__name__)


LOCK_PREFIX = "idempotency:episode:"
DEFAULT_LOCK_TTL = 3600  # 1 hour max execution time


class IdempotencyLock:
    """基于 Redis SET NX EX 的幂等锁。

    Usage:
        lock = IdempotencyLock()
        async with lock.acquire("ep_123"):
            # critical section
    """

    def __init__(self, lock_ttl: int = DEFAULT_LOCK_TTL):
        self._lock_ttl = lock_ttl

    async def try_acquire(self, episode_id: str) -> tuple[bool, Optional[str]]:
        """尝试获取锁。返回 (success, lock_holder_id)。

        Redis 不可用时降级放行：不保证幂等，但允许任务执行。
        避免 Redis 挂了导致 Celery 任务全部静默跳过。
        """
        lock_key = f"{LOCK_PREFIX}{episode_id}"
        holder = str(uuid.uuid4())
        try:
            acquired = await cache.setnx_ex(lock_key, holder, self._lock_ttl)
        except Exception as e:
            logger.warning(
                "Redis unavailable, bypassing idempotency lock for %s: %s",
                episode_id, e,
            )
            # Redis 不可用：放行任务，标记 holder 为 fallback 模式
            return (True, "fallback_no_redis")
        return (acquired, holder if acquired else None)

    async def release(self, episode_id: str, holder: str) -> bool:
        """释放锁（仅锁持有者可释放）。"""
        lock_key = f"{LOCK_PREFIX}{episode_id}"
        current = await cache.get(lock_key)
        if current == holder:
            return await cache.delete(lock_key)
        return False

    async def is_locked(self, episode_id: str) -> bool:
        lock_key = f"{LOCK_PREFIX}{episode_id}"
        return await cache.exists(lock_key)

    async def extend(self, episode_id: str, holder: str) -> bool:
        """续租锁（长时间运行的任务可调用）。"""
        lock_key = f"{LOCK_PREFIX}{episode_id}"
        current = await cache.get(lock_key)
        if current == holder:
            return await cache.set(lock_key, holder, ttl=self._lock_ttl)
        return False

    async def force_release(self, episode_id: str) -> bool:
        """强制释放锁（管理员操作）。"""
        lock_key = f"{LOCK_PREFIX}{episode_id}"
        return await cache.delete(lock_key)


class IdempotencyGuard:
    """幂等守卫：async context manager 模式。

    Usage:
        guard = IdempotencyGuard()
        async with guard("ep_123") as acquired:
            if not acquired:
                raise TaskAlreadyRunning(...)
            # safe execution
    """

    def __init__(self, lock_ttl: int = DEFAULT_LOCK_TTL):
        self._lock = IdempotencyLock(lock_ttl)

    async def __call__(self, episode_id: str):
        return _IdempotencyContext(self._lock, episode_id)


class _IdempotencyContext:
    def __init__(self, lock: IdempotencyLock, episode_id: str):
        self._lock = lock
        self._episode_id = episode_id
        self._holder: Optional[str] = None
        self.acquired = False

    async def __aenter__(self):
        self.acquired, self._holder = await self._lock.try_acquire(self._episode_id)
        return self

    async def __aexit__(self, *args):
        if self.acquired and self._holder:
            await self._lock.release(self._episode_id, self._holder)

    async def extend(self) -> bool:
        if self._holder:
            return await self._lock.extend(self._episode_id, self._holder)
        return False


# 模块级单例
idempotency_lock = IdempotencyLock()
idempotency_guard = IdempotencyGuard()
