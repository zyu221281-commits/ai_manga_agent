"""统一 LLM 调用（DeepSeek + Qwen 双备份 + 缓存装饰器 + 重试）

 依赖收敛策略：
- 主 LLM: DeepSeek--Pro (DeepSeek API)
- 备用 LLM: Qwen3.7-Max (DashScope API)
- 选题分类: Qwen-Turbo (DashScope API)
- 质检: Qwen-VL-Max (DashScope API)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.pricing import calculate_llm_cost


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    finish_reason: str = "stop"
    extra: dict = field(default_factory=dict)


class LLMClient:
    """统一 LLM 客户端，支持 DeepSeek + Qwen 双备份自动切换。

    - deepseek-* 模型走 DeepSeek API (https://api.deepseek.com/v1)
    - qwen-* 模型走 DashScope API (https://dashscope.aliyuncs.com/compatible-mode/v1)
    """

    def __init__(self):
        self._deepseek: Optional[AsyncOpenAI] = None
        self._qwen: Optional[AsyncOpenAI] = None
        self._http = httpx.AsyncClient(timeout=30.0)

    @property
    def deepseek(self) -> AsyncOpenAI:
        if self._deepseek is None:
            self._deepseek = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
            )
        return self._deepseek

    @property
    def qwen(self) -> AsyncOpenAI:
        if self._qwen is None:
            self._qwen = AsyncOpenAI(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._qwen

    async def completions(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek--pro",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        enable_cache: bool = True,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        cache_key = None
        if enable_cache:
            cache_key = self._cache_key(model, messages, temperature)
            cached = await self._get_cache(cache_key)
            if cached:
                cached.pop("from_cache", None)
                return LLMResponse(**cached)

        errors = []
        for attempt_model in self._model_fallback_chain(model):
            try:
                resp = await self._call_openai(
                    attempt_model, messages, temperature, max_tokens, response_format
                )
                if enable_cache and cache_key:
                    await self._set_cache(cache_key, resp)
                return resp
            except Exception as e:
                errors.append("{}: {}".format(attempt_model, e))
                if attempt_model == model:
                    continue
                break

        raise RuntimeError("All LLM backends failed: " + "; ".join(errors))

    async def completions_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek--pro",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[tuple[str, int]]:
        client = self._get_client(model)
        start = time.monotonic()
        try:
            stream = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, stream=True,
            )
            total_output = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    total_output += len(delta)
                    yield delta, 0
        finally:
            (time.monotonic() - start) * 1000

    async def _call_openai(
        self, model: str, messages: list[dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[dict],
    ) -> LLMResponse:
        client = self._get_client(model)
        start = time.monotonic()
        kwargs = dict(model=model, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if response_format:
            kwargs["response_format"] = response_format

        completion = await client.chat.completions.create(**kwargs)
        duration_ms = (time.monotonic() - start) * 1000
        choice = completion.choices[0]
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = calculate_llm_cost(model, input_tokens, output_tokens)

        # DeepSeek reasoner 系列会把内容放到 reasoning_content，content 可能为空
        # 兼容处理：优先 content，为空时回退到 reasoning_content
        msg = choice.message
        content = msg.content or ""
        extra: dict[str, Any] = {}
        reasoning = getattr(msg, "reasoning_content", None)
        if not content and reasoning:
            content = reasoning
            extra["from_reasoning_content"] = True
        if reasoning:
            extra["reasoning_content"] = reasoning

        return LLMResponse(
            content=content, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, duration_ms=duration_ms,
            finish_reason=choice.finish_reason or "stop",
            extra=extra,
        )

    def _get_client(self, model: str) -> AsyncOpenAI:
        if model.startswith("deepseek"):
            return self.deepseek
        return self.qwen

    def _model_fallback_chain(self, primary: str) -> list[str]:
        if primary.startswith("deepseek"):
            return [primary, "qwen3.7-max"]
        if primary.startswith("qwen"):
            return [primary, "deepseek--pro"]
        return [primary]

    @staticmethod
    def _cache_key(model: str, messages: list[dict], temperature: float) -> str:
        raw = json.dumps({"m": model, "msgs": messages, "t": temperature}, sort_keys=True)
        return "llm:cache:" + hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def _get_cache(self, key: str) -> Optional[dict]:
        try:
            from app.services.cache import cache
            data = await cache.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def _set_cache(self, key: str, resp: LLMResponse) -> None:
        try:
            from app.services.cache import cache
            payload = json.dumps({
                "content": resp.content, "model": resp.model,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd, "duration_ms": resp.duration_ms,
                "finish_reason": resp.finish_reason, "from_cache": False,
            })
            await cache.set(key, payload, ttl=7 * 86400)
        except Exception:
            pass

    async def close(self):
        if self._deepseek:
            await self._deepseek.close()
        if self._qwen:
            await self._qwen.close()
        await self._http.aclose()


llm_client = LLMClient()
