"""Agent 基类（含成本上报钩子 +  文件级血缘追踪）

所有 Agent 继承此基类，获得：
- 统一的 LLM 调用接口
- 自动成本记录
- 日志/trace 集成
-  文件级数据血缘追踪（无需 DB）
- 错误处理
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm_client import llm_client, LLMResponse
from app.services.cost_tracker import CostTracker
from app.services.lineage_tracker import LineageTracker
from app.services.file_lineage_tracker import FileLineageTracker


@dataclass
class AgentResult:
    """Agent 执行结果。"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Agent 基类。

    Usage:
        class MyAgent(BaseAgent):
            agent_name = "my_agent"

            async def execute(self, **kwargs) -> AgentResult:
                ...
    """

    agent_name: str = "base"

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        episode_id: Optional[str] = None,
        series_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tracer: Optional[FileLineageTracker] = None,
    ):
        self.session = session
        self.episode_id = episode_id
        self.series_id = series_id
        self.trace_id = trace_id
        self._tracer = tracer  #  文件级血缘追踪器（由外部注入）
        self.logger = logging.getLogger(f"agent.{self.agent_name}")

    @property
    def cost_tracker(self) -> Optional[CostTracker]:
        if self.session:
            return CostTracker(self.session)
        return None

    @property
    def lineage_tracker(self) -> Optional[LineageTracker]:
        """DB 级血缘追踪（需要 session）。"""
        if self.session:
            return LineageTracker(self.session)
        return None

    # ================================================================
    # LLM helpers
    # ================================================================

    async def _llm(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek--pro",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        enable_cache: bool = True,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """统一的 LLM 调用，自动记录成本。"""
        resp = await llm_client.completions(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_cache=enable_cache,
            response_format=response_format,
        )
        if self.cost_tracker:
            await self.cost_tracker.record_llm(
                episode_id=self.episode_id,
                series_id=self.series_id,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                trace_id=self.trace_id,
                operation=self.agent_name,
            )

        # : collect per-call model info for tracer
        if not hasattr(self, '_llm_calls'):
            self._llm_calls: list[dict[str, Any]] = []
        self._llm_calls.append({
            'model': resp.model or '',
            'input_tokens': resp.input_tokens,
            'output_tokens': resp.output_tokens,
            'cost_usd': resp.cost_usd,
            'finish_reason': resp.finish_reason or '',
        })
        return resp

    async def _llm_json(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek--pro",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """JSON 格式 LLM 调用。

        健壮性增强：
        - 优先解析 resp.content
        - 若为空，尝试 resp.extra["from_reasoning_content"] 中的内容
        - 若包含 markdown ```json ... ``` 包裹，自动剥离
        - 若包含其他文本，尝试提取首个 {...} 块
        """
        import json
        import logging

        resp = await self._llm(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.content or ""
        if not content and resp.extra.get("from_reasoning_content"):
            content = resp.extra.get("reasoning_content", "") or ""

        # 直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试剥离 markdown ```json ... ``` 包裹
        stripped = content.strip()
        if stripped.startswith("```"):
            # 去掉首行 ```json / ```，去掉末尾 ```
            lines = stripped.splitlines()
            if len(lines) >= 3:
                inner = "\n".join(lines[1:-1])
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass

        # 尝试提取首个 {...} 块
        first = content.find("{")
        last = content.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(content[first:last + 1])
            except json.JSONDecodeError:
                pass

        logging.getLogger(f"agent.{self.agent_name}").warning(
            "Failed to parse LLM JSON response (len=%d, finish=%s, content_head=%r)",
            len(content), resp.finish_reason, content[:200],
        )
        raise ValueError(
            f"LLM returned non-JSON content (len={len(content)}, finish_reason={resp.finish_reason})"
        )

    # ================================================================
    # Tracking
    # ================================================================

    async def _run_with_tracking(self, **kwargs) -> AgentResult:
        """带追踪的执行包装器（：同时支持 DB lineage + 文件级 lineage）。"""
        start = time.monotonic()

        #  文件级血缘：记录 start
        input_keys = list(kwargs.keys())
        input_shapes = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                input_shapes[k] = f"dict({len(v)} keys)"
            elif isinstance(v, list):
                input_shapes[k] = f"list({len(v)} items)"
            elif isinstance(v, str):
                input_shapes[k] = f"str({len(v)} chars)"
            else:
                input_shapes[k] = type(v).__name__

        if self._tracer:
            self._tracer.record_start(self.agent_name, input_keys=input_keys, input_shapes=input_shapes)

        try:
            self.logger.info("%s started (episode=%s, trace=%s)", self.agent_name, self.episode_id, self.trace_id)
            result = await self.execute(**kwargs)
            duration_ms = (time.monotonic() - start) * 1000
            result.duration_ms = duration_ms

            #  文件级血缘：记录 end
            output_keys = []
            output_shapes = {}
            if result.success and result.data is not None:
                if isinstance(result.data, dict):
                    output_keys = list(result.data.keys())
                    for k, v in result.data.items():
                        if isinstance(v, dict):
                            output_shapes[k] = f"dict({len(v)} keys)"
                        elif isinstance(v, list):
                            output_shapes[k] = f"list({len(v)} items)"
                        elif isinstance(v, str):
                            output_shapes[k] = f"str({len(v)} chars)"
                        else:
                            output_shapes[k] = type(v).__name__

            if self._tracer:
                self._tracer.record_end(
                    agent_name=self.agent_name,
                    output_keys=output_keys,
                    output_shapes=output_shapes,
                    cost_usd=result.cost_usd,
                    duration_ms=duration_ms,
                    success=result.success,
                    error=result.error,
                    model_calls=getattr(self, '_llm_calls', []),
                )

            self.logger.info(
                "%s completed (cost=$%.4f, duration=%.0fms, success=%s)",
                self.agent_name, result.cost_usd, duration_ms, result.success,
            )
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            self.logger.error("%s failed: %s", self.agent_name, str(e)[:200])

            if self._tracer:
                self._tracer.record_end(
                    agent_name=self.agent_name,
                    cost_usd=0.0,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)[:500],
                )

            return AgentResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """子类实现具体逻辑。"""
        ...
