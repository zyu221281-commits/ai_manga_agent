"""Redis 缓存层（LLM/图像/API）

缓存策略（ 文档 2.5）：
- LLM 响应缓存：Redis，TTL 7 天，key = hash(model + messages + temperature)
- 图像生成缓存：MinIO，永久，key = hash(prompt + style + seed)
- 平台 API 响应缓存：Redis，TTL 1 小时
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import redis.asyncio as redis

from app.core.config import settings


class CacheService:
    """Redis 缓存服务，封装常用操作。"""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=20,
            )
        return self._redis

    async def get(self, key: str) -> Optional[str]:
        """获取缓存值。"""
        try:
            r = await self._get_redis()
            return await r.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """设置缓存，带 TTL（秒）。"""
        try:
            r = await self._get_redis()
            await r.set(key, value, ex=ttl)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        try:
            r = await self._get_redis()
            await r.delete(key)
            return True
        except Exception:
            return False

    async def bust_pattern(self, pattern: str) -> int:
        """按模式批量删除缓存键。"""
        try:
            r = await self._get_redis()
            keys = []
            async for key in r.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                await r.delete(*keys)
            return len(keys)
        except Exception:
            return 0

    async def exists(self, key: str) -> bool:
        try:
            r = await self._get_redis()
            return await r.exists(key) > 0
        except Exception:
            return False

    async def setnx_ex(self, key: str, value: str, ttl: int) -> bool:
        """SET NX EX 原子操作（幂等锁）。返回 True 表示获取锁成功。

        注意：本方法不吞异常 — Redis 故障时抛出，由调用方（IdempotencyLock）
        决定是降级放行还是阻塞。这与 LLM 缓存的 fail-soft 语义不同。
        """
        r = await self._get_redis()
        return await r.set(key, value, nx=True, ex=ttl) or False

    async def ping(self) -> bool:
        try:
            r = await self._get_redis()
            return await r.ping()
        except Exception:
            return False

    # ---- LLM 缓存专用 ----

    @staticmethod
    def llm_cache_key(model: str, messages: list[dict], temperature: float) -> str:
        raw = json.dumps({"m": model, "msgs": messages, "t": temperature}, sort_keys=True)
        return f"llm:resp:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

    async def get_llm_cache(self, model: str, messages: list[dict], temperature: float) -> Optional[str]:
        key = self.llm_cache_key(model, messages, temperature)
        return await self.get(key)

    async def set_llm_cache(self, model: str, messages: list[dict], temperature: float, response: str):
        key = self.llm_cache_key(model, messages, temperature)
        await self.set(key, response, ttl=7 * 86400)

    # ---- 图像缓存专用 ----

    @staticmethod
    def image_cache_key(prompt: str, style: str = "", seed: int = 0) -> str:
        raw = f"{prompt}|{style}|{seed}"
        return f"img:cache:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

    async def get_image_cache(self, prompt: str, style: str = "", seed: int = 0) -> Optional[str]:
        key = self.image_cache_key(prompt, style, seed)
        return await self.get(key)

    async def set_image_cache(self, prompt: str, style: str, seed: int, minio_path: str):
        key = self.image_cache_key(prompt, style, seed)
        await self.set(key, minio_path, ttl=30 * 86400)  # 30 days

    # ---- API 响应缓存 ----

    async def get_api_cache(self, url: str) -> Optional[str]:
        key = f"api:resp:{hashlib.md5(url.encode()).hexdigest()[:16]}"
        return await self.get(key)

    async def set_api_cache(self, url: str, response: str, ttl: int = 3600):
        key = f"api:resp:{hashlib.md5(url.encode()).hexdigest()[:16]}"
        await self.set(key, response, ttl=ttl)


# 模块级单例
cache = CacheService()
