"""LLM 适配器：DeepSeek / Qwen 统一接口

切换 provider 只改配置，不改代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    """LLM 统一接口。"""

    @abstractmethod
    async def completions(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def completions_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        ...


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek--Pro 适配器。"""

    @property
    def MODEL(self) -> str:
        """从 config 读取 LLM 模型标识（避免硬编码）。"""
        from app.core.config import settings
        return settings.ARK_LLM_MODEL

    async def completions(self, messages, temperature=0.7, max_tokens=4096):
        from app.services.llm_client import llm_client
        resp = await llm_client.completions(
            messages=messages,
            model=self.MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"content": resp.content, "model": resp.model, "tokens": resp.output_tokens}

    async def completions_json(self, messages, temperature=0.3, max_tokens=4096):
        from app.services.llm_client import llm_client
        import json
        resp = await llm_client.completions(
            messages=messages,
            model=self.MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return {"content": resp.content, "model": resp.model, "json": json.loads(resp.content)}


class QwenAdapter(LLMAdapter):
    """Qwen3.7-Max 适配器。"""

    MODEL = "qwen3.7-max"

    async def completions(self, messages, temperature=0.7, max_tokens=4096):
        from app.services.llm_client import llm_client
        resp = await llm_client.completions(
            messages=messages,
            model=self.MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"content": resp.content, "model": resp.model, "tokens": resp.output_tokens}

    async def completions_json(self, messages, temperature=0.3, max_tokens=4096):
        from app.services.llm_client import llm_client
        import json
        resp = await llm_client.completions(
            messages=messages,
            model=self.MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return {"content": resp.content, "model": resp.model, "json": json.loads(resp.content)}


def get_llm_adapter(provider: str = "deepseek") -> LLMAdapter:
    """工厂方法：根据 provider 名称获取适配器。"""
    adapters = {
        "deepseek": DeepSeekAdapter(),
        "qwen": QwenAdapter(),
    }
    return adapters.get(provider.lower(), DeepSeekAdapter())
