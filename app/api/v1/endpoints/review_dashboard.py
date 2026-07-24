"""人工审核看板（Critic 临界区主动推送）

 审核看板：
- 推送到看板 + 钉钉提醒
- SLA：临界区任务 4h 内必须决策，超时进 DLQ
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ReviewDashboard:
    """人工审核看板管理器。

    Usage:
        dashboard = ReviewDashboard(session)
        await dashboard.push(episode_id="ep_1", source="critic", score=0.75, reason="...")
        pending = await dashboard.get_pending()
    """

    SLA_HOURS = 4

    def __init__(self, session: AsyncSession):
        self._session = session

    async def push(
        self,
        episode_id: str,
        source: str,  # "critic" or "dlq"
        critic_score: Optional[float] = None,
        reason: str = "",
    ) -> str:
        """推送任务到人工审核看板。

        Args:
            episode_id: 剧集 ID
            source: 来源（critic 临界区 / dlq）
            critic_score: Critic 评分
            reason: 推送原因
        """
        import uuid
        review_id = str(uuid.uuid4())
        sla_deadline = datetime.now(timezone.utc) + timedelta(hours=self.SLA_HOURS)

        sql = text("""
            INSERT INTO pending_reviews (id, episode_id, source, critic_score, reason, status, sla_deadline)
            VALUES (:id, :episode_id, :source, :critic_score, :reason, 'pending', :sla_deadline)
        """)
        await self._session.execute(sql, {
            "id": review_id,
            "episode_id": episode_id,
            "source": source,
            "critic_score": critic_score,
            "reason": reason[:1000],
            "sla_deadline": sla_deadline,
        })
        await self._session.commit()

        # 发送通知
        await self._notify(episode_id, source, critic_score, reason)

        logger.info("Review %s pushed to dashboard (source=%s, score=%.2f)", review_id, source, critic_score)
        return review_id

    async def get_pending(self, source: str = "") -> list[dict]:
        """获取待审核列表。"""
        where = ""
        params = {}
        if source:
            where = "WHERE source = :source"
            params["source"] = source

        sql = text(f"""
            SELECT * FROM pending_reviews {where}
            ORDER BY
                CASE WHEN sla_deadline < NOW() THEN 0 ELSE 1 END,
                created_at
            LIMIT 50
        """)
        result = await self._session.execute(sql, params)
        return [dict(row._mapping) for row in result]

    async def decide(
        self,
        review_id: str,
        decision: str,  # "approve" / "reject" / "edit"
        decided_by: str = "admin",
        comment: str = "",
    ) -> bool:
        """审核决策。"""
        sql = text("""
            UPDATE pending_reviews
            SET status = 'decided', decision = :decision, decided_by = :decided_by, decided_at = NOW()
            WHERE id = :id
        """)
        result = await self._session.execute(sql, {
            "id": review_id,
            "decision": decision,
            "decided_by": decided_by,
        })
        await self._session.commit()

        if result.rowcount == 0:
            logger.warning("Review %s not found or already decided", review_id)
            return False

        logger.info("Review %s decided: %s by %s", review_id, decision, decided_by)
        return True

    async def check_sla_breaches(self) -> list[dict]:
        """检查 4h SLA 超时未决策的任务。"""
        sql = text("""
            SELECT * FROM pending_reviews
            WHERE status = 'pending' AND sla_deadline < NOW()
        """)
        result = await self._session.execute(sql)
        breaches = [dict(row._mapping) for row in result]

        for b in breaches:
            logger.warning("SLA breach: review %s for episode %s", b["id"], b["episode_id"])

        return breaches

    async def get_stats(self) -> dict:
        """看板统计。"""
        sql = text("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                COUNT(*) FILTER (WHERE status = 'pending' AND sla_deadline < NOW()) AS sla_breached,
                COUNT(*) FILTER (WHERE status = 'decided') AS decided
            FROM pending_reviews
        """)
        result = await self._session.execute(sql)
        row = result.fetchone()
        return {
            "pending": row.pending if row else 0,
            "sla_breached": row.sla_breached if row else 0,
            "decided": row.decided if row else 0,
        }

    async def _notify(
        self,
        episode_id: str,
        source: str,
        score: Optional[float],
        reason: str,
    ):
        """发送钉钉/飞书通知。"""
        score_str = f"{score:.2f}" if score is not None else "N/A"
        message = f"新审核任务 | 剧集: {episode_id} | 来源: {source} | 评分: {score_str}"
        logger.info("Review notification: %s", message)
        # 在实际项目中调用 AlertManager
