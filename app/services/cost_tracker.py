"""成本计量中间件（所有调用必经）

 成本可观测 + 软告警设计：
- 记录所有 LLM/图像/视频/TTS/审核调用的成本到 PG cost_ledger
- 提供日/月/单集预算查询
- Demo 模式仅告警不熔断（production 模式可开启硬止损）
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.pricing import calculate_llm_cost, calculate_unit_cost

logger = logging.getLogger(__name__)


class CostTracker:
    """成本计量中间件。

    用法:
        tracker = CostTracker(session)
        await tracker.record_llm(episode_id, "deepseek--pro", 1000, 500)
        remaining = await tracker.query_remaining_daily()
    """

    def __init__(self, session: AsyncSession):
        self._session = session
        self._alerted_80 = False

    # ================================================================
    # Record
    # ================================================================

    async def record_llm(
        self,
        episode_id: Optional[str],
        model: str,
        input_tokens: int,
        output_tokens: int,
        series_id: Optional[str] = None,
        operation: str = "llm_call",
        trace_id: Optional[str] = None,
    ) -> float:
        """记录 LLM 成本。返回 cost_usd。

        PG 不可用时静默降级：成本记录丢失，但不影响主管线。
        BudgetHardStopError（生产环境硬止损）不吞，正常透传。
        """
        cost = calculate_llm_cost(model, input_tokens, output_tokens)
        try:
            await self._insert_record(
                episode_id=episode_id,
                series_id=series_id,
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                trace_id=trace_id,
            )
            await self._check_alerts()
        except Exception as e:
            # 生产环境硬止损必须透传，不能被 PG 故障吞掉
            from app.core.exceptions import BudgetHardStopError
            if isinstance(e, BudgetHardStopError):
                raise
            logger.warning(
                "Failed to record LLM cost to PG (ep=%s, model=%s, cost=$%.4f): %s",
                episode_id, model, cost, e,
            )
        return cost

    async def record_unit(
        self,
        episode_id: Optional[str],
        model: str,
        operation: str,
        unit_count: float,
        series_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> float:
        """记录按次/按量定价成本（图像/视频/TTS/审核）。返回 cost_usd。

        PG 不可用时静默降级：成本记录丢失，但不影响主管线。
        BudgetHardStopError（生产环境硬止损）不吞，正常透传。
        """
        cost = calculate_unit_cost(model, unit_count)
        try:
            await self._insert_record(
                episode_id=episode_id,
                series_id=series_id,
                model=model,
                operation=operation,
                unit_count=unit_count,
                cost_usd=cost,
                trace_id=trace_id,
            )
            await self._check_alerts()
        except Exception as e:
            from app.core.exceptions import BudgetHardStopError
            if isinstance(e, BudgetHardStopError):
                raise
            logger.warning(
                "Failed to record unit cost to PG (ep=%s, model=%s, op=%s, cost=$%.4f): %s",
                episode_id, model, operation, cost, e,
            )
        return cost

    async def _insert_record(self, **kwargs):
        cols = ["id"] + [k for k in kwargs if kwargs[k] is not None]
        vals = [f"'{uuid.uuid4()}'"] + [f":{k}" for k in cols[1:]]
        sql = f"INSERT INTO cost_ledger ({', '.join(cols)}) VALUES ({', '.join(vals)})"
        params = {k: kwargs[k] for k in cols[1:]}
        await self._session.execute(text(sql), params)
        await self._session.commit()

    # ================================================================
    # Query
    # ================================================================

    async def query_remaining_daily(self) -> float:
        """查询当日剩余预算（USD）。"""
        used = await self._daily_used()
        return settings.COST_DAILY_CAP_USD - used

    async def query_remaining_monthly(self) -> float:
        """查询当月剩余预算（USD）。"""
        used = await self._monthly_used()
        return settings.COST_MONTHLY_CAP_USD - used

    async def daily_used_ratio(self) -> float:
        used = await self._daily_used()
        return used / settings.COST_DAILY_CAP_USD if settings.COST_DAILY_CAP_USD > 0 else 0.0

    async def monthly_used_ratio(self) -> float:
        used = await self._monthly_used()
        return used / settings.COST_MONTHLY_CAP_USD if settings.COST_MONTHLY_CAP_USD > 0 else 0.0

    async def episode_cost(self, episode_id: str) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger WHERE episode_id = :eid"
        result = await self._session.execute(text(sql), {"eid": episode_id})
        return float(result.scalar_one())

    async def series_cost(self, series_id: str) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger WHERE series_id = :sid"
        result = await self._session.execute(text(sql), {"sid": series_id})
        return float(result.scalar_one())

    async def _daily_used(self) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger WHERE created_at >= CURRENT_DATE"
        result = await self._session.execute(text(sql))
        return float(result.scalar_one())

    async def _monthly_used(self) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger WHERE date_trunc('month', created_at) = date_trunc('month', CURRENT_DATE)"
        result = await self._session.execute(text(sql))
        return float(result.scalar_one())

    # ================================================================
    # Alerts
    # ================================================================

    async def _check_alerts(self):
        ratio = await self.daily_used_ratio()
        if ratio >= settings.COST_HARD_STOP_THRESHOLD and settings.cost_hard_stop_enabled:
            from app.core.exceptions import BudgetHardStopError
            raise BudgetHardStopError(f"Daily budget 100% exhausted ({ratio:.0%})")
        if ratio >= settings.COST_ALERT_THRESHOLD and not self._alerted_80:
            self._alerted_80 = True
            # 在实际项目中这里发送钉钉/飞书告警
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Daily budget alert: {ratio:.0%} used ({settings.COST_ALERT_THRESHOLD:.0%} threshold)")

    # ================================================================
    # Dashboard
    # ================================================================

    async def dashboard(self) -> dict:
        """返回预算看板数据。"""
        return {
            "daily_used": await self._daily_used(),
            "daily_cap": settings.COST_DAILY_CAP_USD,
            "daily_ratio": await self.daily_used_ratio(),
            "monthly_used": await self._monthly_used(),
            "monthly_cap": settings.COST_MONTHLY_CAP_USD,
            "monthly_ratio": await self.monthly_used_ratio(),
            "per_episode_budget": settings.COST_BUDGET_PER_EPISODE_USD,
        }
